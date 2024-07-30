import math
import random
import numpy as np
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset.dataset import PreTrainingDataset, OfflineDataset

import os
import os.path as opt
import sys
sys.path.append('./model')
from model.stmae import STMAE, Encoder
from utils.stmae_utils import train_epoch, valid_epoch, stmae_training_loop
from model.stgcn import STGCN
from utils.stgcn_utils import train_one_epoch, valid_one_epoch, stgcn_offline_training_loop

from anatomical_loss import AnatomicalLoss, get_data_stats, plot_data_stats


def seed_everything(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train STMAE')
	parser.add_argument('--cfg_path', default='configs/train_STMAE.yaml', help='Path to the train.yaml config')
	args = parser.parse_args()
	## configs
	cfg_path = args.cfg_path
	args = OmegaConf.load(cfg_path)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	seed_everything(args.seed)

	print('\n\n', '='*15, 'ARGUMENTS.', '='*15)
	for arg, val in args.__dict__['_content'].items():
		if isinstance(val, omegaconf.nodes.AnyNode):
			print('> {}: {}'.format(arg, val))
		else:
			print('> {}'.format(arg))
			for arg_, val_ in val.items():
				print('\t- {}: {}'.format(arg_, val_))


	## FIRST PHASE: PRE-TRAINING
	print('\n\n', '='*15, 'FIRST PHASE: PRE-TRAINING', '='*15)

	## DATA SETS & LOADERS
	print('\nLOADING DATA....')
	data_args = args.data
	stmae_args = args.stmae

	info = {
		'dataset': args.dataset,
		'n_joints': data_args.n_joints,
		'mean': data_args.mean,
		'std': data_args.std,
		'joints_connections': data_args.joints_connections,
		'label_map': data_args.label_map,
	}

	train_set = PreTrainingDataset(data_dir=data_args.train_data_dir,
									 window_size=stmae_args.window_size,
									 step=data_args.step,
									 normalize=data_args.normalize,
									 info=info,
									 random_rot=False)

	valid_set = PreTrainingDataset(data_dir=data_args.test_data_dir,
									 window_size=stmae_args.window_size,
									 step=data_args.step,
									 normalize=data_args.normalize,
									 info=info,	
									 )

	print('# Train: {}, # Valid: {}'.format(len(train_set), len(valid_set)))
	train_loader = DataLoader(train_set, batch_size=stmae_args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=stmae_args.batch_size, shuffle=False)

	print('\nBUILDING MODEL....')
	## MODEL & OPTIMIZER
	if stmae_args.anatomical_loss:

		stats_set = PreTrainingDataset(data_dir=data_args.train_data_dir,
									 window_size=stmae_args.window_size,
									 step=stmae_args.window_size, ## overlook redondante frames
									 normalize=data_args.normalize,
									 info=info)
		stats_loader = DataLoader(stats_set, batch_size=stmae_args.batch_size, shuffle=True)

		angles_stats, lengths_stats = get_data_stats(stats_loader, stmae_args.root_index)
		os.makedirs(opt.join(args.save_folder_path, args.dataset, args.exp_name), exist_ok=True)
		plot_data_stats(angles_stats, lengths_stats, 
						fname=opt.join(args.save_folder_path, args.dataset, args.exp_name, 'stats.png'))
		anatomical_loss = AnatomicalLoss(angles_stats['min'].to(device),
										 angles_stats['max'].to(device),
										 lengths_stats['min'].to(device), 
										 lengths_stats['max'].to(device), 
										 stmae_args.root_index)
	else:
		anatomical_loss = None

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
				anatomical_loss=anatomical_loss	
			).to(device)

	optimizer = optim.AdamW(stmae.parameters(), lr=stmae_args.lr)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

	n_params = sum(p.numel() for p in stmae.parameters() if p.requires_grad)
	print("Number of trainable parameters of STMAE: ", n_params)

	## /!\ if the stmae model is already pre-trained, comment this line
	stmae_training_loop(stmae, train_loader, valid_loader, device, optimizer, scheduler, args)



	## SECOND PHASE: FINE-TUNING
	print('\n\n', '='*15, 'SECOND PHASE: FINE-TUNING', '='*15)

	## DATA SETS & LOADERS
	print('\nLOADING DATA....')
	data_args = args.data
	stmae_args = args.stmae

	stgcn_args = args.stgcn_offline
	train_set = OfflineDataset(data_dir=data_args.train_data_dir,
								sequence_length=stgcn_args.sequence_length,
								info=info,
								normalize=data_args.normalize) 

	valid_set = OfflineDataset(data_dir=data_args.test_data_dir,
								sequence_length=stgcn_args.sequence_length,
								info=info,
								normalize=data_args.normalize)

	print('# Train: {}, # Valid: {}'.format(len(train_set), len(valid_set)))
	train_loader = DataLoader(train_set, batch_size=stgcn_args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=stgcn_args.batch_size, shuffle=False)


	print('\nLOADING STMAE PRE-TRAINED WEIGHTS....', end='')	
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
		
	chkpt = torch.load(stmae_chkpt)
	stmae.load_state_dict(chkpt['state_dict'])
	stmae = stmae.to(device)
	print('[optimal epoch={}]'.format(chkpt['epoch']))

	print('\nBUILDING STGCN MODEL....')
	graph_cfg = dict(layout=args.dataset,
					 mode='spatial')
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
				 ).to(device)

	n_params = sum(p.numel() for p in stgcn.parameters() if p.requires_grad)
	print("Number of trainable parameters of STGCN: ", n_params)


	optimizer = optim.AdamW(stgcn.parameters(), lr=stgcn_args.lr)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
	criterion = nn.CrossEntropyLoss()

	stgcn_offline_training_loop(stgcn, stmae, train_loader, valid_loader, device, optimizer, criterion, scheduler, args)


	## THIRD PHASE: EVALUATION
	print('\n\n', '='*15, 'THIRD PHASE: EVALUATION', '='*15)
	print('\nLOADING STGCN PRE-TRAINED WEIGHTS....')
	stgcn_args = args.stgcn_offline
	graph_cfg = dict(layout=args.dataset,
					 mode='spatial')
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
				 ).to(device)

	stgcn_chkpt = opt.join(args.save_folder_path, args.dataset, args.exp_name,'weights', 'best_offline_stgcn_model.pth')
	if not os.path.isfile(stgcn_chkpt):
		print(f"File not found: ", stgcn_chkpt)
		raise FileNotFoundError

	chkpt = torch.load(stgcn_chkpt)
	stgcn.load_state_dict(chkpt['state_dict'])
	stgcn = stgcn.to(device)
	epoch = chkpt['epoch']

	valid_loss, accuracy, true_labels, pred_labels = valid_one_epoch(stgcn, stmae, 
																	valid_loader, 
																	criterion, device)

	save_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name, 'confusion_matrix/')
	fig, ax = plt.subplots(figsize=(10, 10))
	sns.heatmap(confusion_matrix(true_labels, pred_labels, normalize='true')*100,
				ax=ax, annot=True, fmt='.1f', 
				xticklabels=valid_loader.dataset.label_map, yticklabels=valid_loader.dataset.label_map)
	ax.set_title('Confusion Matrix (acc={})'.format(accuracy))
	plt.gcf().savefig(opt.join(save_folder_path, 'cm_best.png'))