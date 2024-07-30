import math
import random
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.dataset import OnlineEvalDataset

import os
import os.path as opt
import sys
sys.path.append('./model')
from model.stmae import STMAE, Encoder

from utils.visualize import display_sample



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Demo STMAE')
	parser.add_argument('--cfg_path', default='configs/train_STMAE.yaml', help='Path to the train.yaml config')
	parser.add_argument('--file_idx', help='sequence index to display')
	args = parser.parse_args()
	## configs
	cfg_path = args.cfg_path
	file_idx = args.file_idx
	args = OmegaConf.load(cfg_path)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print('LOADING DATA....')
	data_args = args.data
	model_args = args.stmae
	data_set = OnlineEvalDataset(data_dir=data_args.test_data_dir,
								file_idx=file_idx,
								window_size=model_args.window_size,
								step=data_args.step,
								normalize=data_args.normalize)

	data_loader = DataLoader(data_set, batch_size=128, shuffle=True)
	print()

	print('BUILDING MODEL....')
	## MODEL & OPTIMIZER
	stmae_args = args.stmae
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

	print('\nDISPLAYING....')
	out = next(iter(data_loader))
	i = random.randint(0 ,len(out['Sequence']))
	display_sample(stmae, out['Sequence'][i:i+1].to(device), 
					data_set.joints_connections,
					fname=None)

