import math
import random
import numpy as np
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import argparse
import time 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat
import torch.optim as optim

from dataset.dataset import OnlineEvalDataset

import matplotlib.pyplot as plt

import os
import os.path as opt
import sys
sys.path.append('./model')
from model.vit_mae import ViT
from model.stmae import STMAE, Encoder
from model.stgcn import STGCN
from utils.stgcn_utils import get_best_detection_threshold
from utils.postprocessing import gaussian_smooth, segment_labels_sequence, postprocess, pad_sequence, levenshtein_accuracy


from scipy import stats as st


def seed_everything(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	

def eval_loop(stmae, stgcn, device, args):

	stmae.eval()
	stgcn.eval()

	txt_save_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name, 'results/')
	img_save_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name, 'predictions/')
	os.makedirs(txt_save_folder_path, exist_ok=True)
	os.makedirs(img_save_folder_path, exist_ok=True)

	sequences_names = []
	with open(opt.join(args.data.test_data_dir, "annotations.txt"), "r") as gt:
		for line in tqdm(gt.readlines()):
			line = line.split(";")
			sequences_names.append(line[0])
				 
	output_file_content = ''
	pbar = tqdm(sequences_names)

	anim_symbs = ['|', '/', '-', '\\', '-']

	accs = []
	times = []
	cnt = 0
	stgcn_on_args = args.stgcn_online
	stgcn_off_args = args.stgcn_offline

	levenshtein_accs = []

	for file in pbar:
		
		output_file_content += f"{file};"
		
		test_set = OnlineEvalDataset(
			data_dir=args.data.test_data_dir,
			file_idx=file,
			window_size=stgcn_on_args.window_size,
			step=args.data.step,
			normalize=args.data.normalize,
		)

		test_loader = DataLoader(test_set, batch_size=stgcn_on_args.batch_size, shuffle=False)

		predictions = [test_set.non_gesture_idx] * (stgcn_on_args.window_size // 2)
		probabilities = [0.0] * (stgcn_on_args.window_size // 2)

		cnt = 0
		with torch.no_grad():
			for d in test_loader:
				desc = f"> {anim_symbs[cnt % len(anim_symbs)]} online processing: file {file} [{cnt+1}/{len(test_loader)}]...."
				pbar.set_description(desc)

				seq = d['Sequence'].to(device)
				labels = d['Sequence'].to(device)

				start_time = time.time()
				enc_seq = stmae.inference(seq)
				gesture_proba, type_proba, _, _ = stgcn(enc_seq)
				inference_time = (time.time() - start_time) / (len(enc_seq) * len(enc_seq[0])) # batch size * num_frames
				times.append(inference_time)

				pred_label = gesture_proba.argmax(dim=-1).detach().cpu().numpy().flatten()
				labels = labels.detach().cpu().numpy().flatten()
				type_proba = type_proba.detach().cpu().numpy().flatten()
				predictions = np.concatenate([predictions, pred_label], axis=0)
				probabilities = np.concatenate([probabilities, type_proba], axis=0)

				cnt += 1

		predictions = np.concatenate(
			[predictions, np.array([test_set.non_gesture_idx] * ((stgcn_on_args.window_size // 2) + stgcn_on_args.window_size % 2))], axis=0
		)

		probabilities = np.concatenate(
			[probabilities, np.array([0.0] * ((stgcn_on_args.window_size // 2) + stgcn_on_args.window_size % 2))], axis=0
		)


		thr_fname = opt.join(img_save_folder_path, str(file)+'_roc.png')
		try: 
			optimal_threshold, roc = get_best_detection_threshold(test_set.gt_window_gesture, probabilities, fname=thr_fname)
		except Exception as err:
			optimal_threshold = 0.8


		predictions = postprocess(predictions, test_set.non_gesture_idx, stgcn_on_args.min_seq_length, stgcn_on_args.max_seq_length)
		predictions_seg = segment_labels_sequence(predictions)

		sequence = test_set.sequence_data.unsqueeze(0).to(device)

		for (label, start, end) in predictions_seg:
			if label == test_set.non_gesture_idx or end <= start:
				continue

			output_file_content += "{};{};{};".format(test_set.label_map[int(label)], start, end)

		fig, ax = plt.subplots(figsize=(10, 4))
		ax.plot(predictions, 'r--', label='predictions')
		ax.plot(probabilities, 'k:', label='probabilities')
		ax.plot(test_set.gt_window, 'g-.', label='true')
		ax.set_title('Sequence Gesture')
		plt.legend()
		fig.tight_layout()
		fig.savefig(opt.join(img_save_folder_path, str(file)+'.png'))
		plt.close()
		
		output_file_content += "\n"
		
		## updating file while looping, to check in real-time
		with open(opt.join(txt_save_folder_path, args.output_file_name), "w") as f:
			f.write(output_file_content)

	with open(opt.join(txt_save_folder_path, args.output_file_name), "w") as f:
		f.write(output_file_content)

	print('avg time (ms):', np.mean(times))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train STMAE Online')
	parser.add_argument('--cfg_path', default='configs/shrec21.yaml', help='Path to the train.yaml config')
	args = parser.parse_args()
	## configs
	cfg_path = args.cfg_path
	args = OmegaConf.load(cfg_path)

	device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
	seed_everything(args.seed)

	print('\n\n', '='*15, 'ARGUMENTS.', '='*15)
	for arg, val in args.__dict__['_content'].items():
		if isinstance(val, omegaconf.nodes.AnyNode):
			print('> {}: {}'.format(arg, val))
		else:
			print('> {}'.format(arg))
			for arg_, val_ in val.items():
				print('\t- {}: {}'.format(arg_, val_))
	
	print('\nLOADING STMAE PRE-TRAINED WEIGHTS....', end='')	
	stmae_args = args.stmae
	## MODEL & OPTIMIZER
	encoder = Encoder(
				patch_num=stmae_args.num_joints*stmae_args.window_size,
				patch_dim=stmae_args.coords_dim,
				window_size=stmae_args.window_size,
				num_classes=stmae_args.coords_dim,
				dim=stmae_args.encoder_embed_dim,
				depth=stmae_args.encoder_depth,
				heads=stmae_args.num_heads,
				mlp_dim=stmae_args.mlp_dim ,
				pool = 'cls',
				# channels = 3,
				dim_head = 64,
				dropout = 0.,
				emb_dropout = 0.
			)

	stmae = STMAE(
				encoder=encoder,
				decoder_dim=stmae_args.decoder_dim,
				decoder_depth=stmae_args.decoder_depth,
				masking_strategy=stmae_args.masking_strategy,
				spatial_masking_ratio=stmae_args.spatial_masking_ratio,
				temporal_masking_ratio=stmae_args.temporal_masking_ratio,
			)
	
	stmae_chkpt = opt.join(args.save_folder_path, args.dataset, args.exp_name,'weights', 'best_stmae_model.pth')
	if not opt.isfile(stmae_chkpt):
		print(f"File not found: ", stmae_chkpt)
		raise FileNotFoundError

	stmae_chkpt = torch.load(stmae_chkpt)
	stmae.load_state_dict(stmae_chkpt['state_dict'])
	stmae = stmae.to(device)
	print('[optimal epoch={}]'.format(stmae_chkpt['epoch']))


	print('\nLOADING ONLINE STGCN PRE-TRAINED WEIGHTS....', end='')
	graph_cfg = dict(layout=args.dataset,
					 mode='spatial')
	stgcn_args = args.stgcn_online
	stgcn = STGCN(graph_cfg,
				  in_channels=stmae_args.decoder_dim,
				  base_channels=64,
				  ch_ratio=2,
				  num_stages=6,
				  inflate_stages=[3, 5],
				  down_stages=[3, 5],
				  pretrained=None,
				 gcn_with_res=False,
				 task=stgcn_args.task,
				 tcn_type='unit_tcn',
				 num_classes=stgcn_args.num_classes,
				 num_gesture_classes=1,
				 )

	stgcn_chkpt = opt.join(args.save_folder_path, args.dataset, args.exp_name,'weights', 'best_online_stgcn_model.pth')
	if not os.path.isfile(stgcn_chkpt):
		print(f"File not found: ", stgcn_chkpt)
		raise FileNotFoundError

	stgcn_chkpt = torch.load(stgcn_chkpt)
	stgcn.load_state_dict(stgcn_chkpt['state_dict'])
	stgcn = stgcn.to(device)
	print('[optimal epoch={}]'.format(stgcn_chkpt['epoch']))

	args.output_file_name = 'online_evaluation_results.txt'

	print('\nINFERENCE....\n')
	eval_loop(stmae, stgcn, device, args)