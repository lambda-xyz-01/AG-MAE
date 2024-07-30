import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm
import glob
import sys
import os.path as opt
from omegaconf import OmegaConf


def random_rotation(skel, ang):
	cos_ang = torch.cos(ang)
	sin_ang = torch.sin(ang)
	rot_mat = torch.tensor([[cos_ang, -sin_ang],
							[sin_ang, cos_ang]]).float()
	skel[..., :2] = torch.matmul(skel[..., :2], rot_mat)
	return skel


class PreTrainingDataset(Dataset):
	def __init__(
		self,
		data_dir,
		window_size,
		step,
		info,
		normalize=False,
		random_rot=False
	):
		"""
		Parameters:
		data_dir (str): path to the dataset directory
		window_size (int): sliding window size
		step (int): stride for sliding window
		normalize (bool): to normalize data using train split mean and variance

		mean (list): list of 3 floats indicating the mean of each coordinates (x, y, z). 
		std (list): list of 3 floats indicating the std of each coordinates (x, y, z). 

		/!\\ => mean & std Should be provided if normalize is True
		"""
		self.data_dir = data_dir
		self.window_size = window_size
		self.step = step
		self.normalize = normalize
		self.random_rot = random_rot

		self.dataset_name = info['dataset']
		if normalize:
			mean = torch.tensor(info['mean'], dtype=torch.float64)
			std = torch.tensor(info['std'], dtype=torch.float64)
			self.mean = mean.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords
			self.std = std.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords

		self.label_map = info['label_map']
		self.joints_connections = info['joints_connections']
		self.n_joints = info['n_joints']
		self.load_data()

	def load_data(self):

		self.sequences = []

		with open(opt.join(self.data_dir, "annotations.txt"), "r") as annotations:
			## annotations should follow this format
			# sequence_Id;gesture_1;start_frame;end_frame;...;gesture_n;start_frame;end_frame;"
			# ...
			pbar = tqdm(annotations.readlines(), desc="Loading Pre-Training {} Dataset....".format(self.dataset_name), colour='green')
			for line in pbar:
				line = line.split(";")
				file_name = line[0]
				data = line[1:-1]

				sequence_data = []

				with open(opt.join(self.data_dir, f"{file_name}.txt"), "r") as fd:
					for line_idx, line_ in enumerate(fd.readlines()):
						line_ = line_.split(";")[1:-1]  # remove index and end-of-line
						line_ = np.reshape(line_, (self.n_joints, 3)).astype(np.float64)  
						sequence_data.append(line_)

				sequence_data = np.array(sequence_data).astype(np.float64)
				# generate windows of size W for current gesture
				for cursor in range(0, sequence_data.shape[0] - self.window_size, self.step):
					self.sequences.append(sequence_data[cursor : cursor + self.window_size, :, :])

					
	def __len__(self):
		return len(self.sequences)

	def _normalize(self, x):
		'''
		normalize data with train mean and variance 
		'''
		return (x - self.mean) / self.std

	def _preprocess(self, sequence: np.ndarray):
		'''
		preprocessing the sequences
		'''
		sequence_tensor = torch.from_numpy(sequence).float()
		if self.normalize:
			sequence_tensor = self._normalize(sequence_tensor)
		return sequence_tensor

	def __getitem__(self, index):
		sequence = self.sequences[index]
		sequence = self._preprocess(sequence)
		if self.random_rot:
			sequence = random_rotation(sequence, torch.randint(0, 360, size=(1,)))

		return dict(Sequence=sequence)


class OfflineDataset(Dataset):
	def __init__(
		self,
		data_dir,
		sequence_length,
		normalize=False,
	):
		"""
		Parameters:
		data_dir (str): path to the dataset directory
		sequence_length (int): sequence length
		step (int): stride for sliding window
		normalize (bool): to normalize data using train split mean and variance
		mean (list): list of 3 floats indicating the mean of each coordinates (x, y, z). 
		std (list): list of 3 floats indicating the std of each coordinates (x, y, z). 

		/!\\ => mean & std Should be provided if normalize is True
		"""
		self.data_dir = data_dir
		self.sequence_length = sequence_length
		self.normalize = normalize

		self.dataset_name = info['dataset']
		if normalize:
			mean = torch.tensor(info['mean'], dtype=torch.float64)
			std = torch.tensor(info['std'], dtype=torch.float64)
			self.mean = mean.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords
			self.std = std.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords

		self.label_map = info['label_map']
		self.joints_connections = info['joints_connections']
		self.n_joints = info['n_joints']
		self.load_data()

	def load_data(self):

		self.sequences = []
		self.labels = []

		with open(opt.join(self.data_dir, "annotations.txt"), "r") as annotations:
			## annotations should follow this format
			# sequence_Id;gesture_1;start_frame;end_frame;...;gesture_n;start_frame;end_frame;"
			# ...
			pbar = tqdm(annotations.readlines(), desc="Loading Offline {} Dataset....".format(self.dataset_name), colour='green')
			for line in pbar:
				line = line.split(";")
				file_name = line[0]
				line = line[1:-1]

				sequence_data = []
				with open(opt.join(self.data_dir, f"{file_name}.txt"), "r") as fd:
					for line_idx, seq_line in enumerate(fd.readlines()):
						seq_line = seq_line.split(";")[1:-1]  # remove index and end-of-line
						seq_line = np.reshape(seq_line, (self.n_joints, 3)).astype(np.float64)  
						sequence_data.append(seq_line)

				sequence_data = np.array(sequence_data).astype(np.float64)

				if len(sequence_data) == 0:
					continue

				if len(line) > 1:
					# for each gesture
					for index in range(0, len(line), 3):
						# gesture start & end frames
						s = int(line[index + 1])
						e = int(line[index + 2])
						if e <= s:
							continue
						# gesture label
						lab = line[index]
						self.sequences.append(sequence_data[s:e+1])
						self.labels.append(self.label_map.index(lab))
				else:
					self.sequences.append(sequence_data)
					self.labels.append(self.label_map.index(line[-1]))
					
	def __len__(self):
		return len(self.sequences)

	def _normalize(self, x):
		'''
		normalize data with train mean and variance 
		'''
		return (x - self.mean) / self.std

	def pad_sequence(self, seq, target_length):
		seq_len = seq.shape[0]
		if seq_len >= target_length:
			return seq[:target_length]
		pad_len = target_length - seq_len
		padding = torch.zeros((pad_len, *seq.shape[1:]), dtype=seq.dtype)
		return torch.cat([seq, padding], dim=0)

	def _preprocess(self, sequence: np.ndarray):
		'''
		preprocessing the sequences
		'''
		sequence_tensor = torch.from_numpy(sequence).float()
		if self.normalize:
			sequence_tensor = self._normalize(sequence_tensor)
		sequence_tensor = self.pad_sequence(sequence_tensor, self.sequence_length)
		return sequence_tensor

	def __getitem__(self, index):
		sequence = self.sequences[index]
		sequence = self._preprocess(sequence)
		label = self.labels[index]

		return dict(Sequence=sequence, 
					Label=label,
					)



class OnlineTrainingDataset(Dataset):
	def __init__(
		self,
		data_dir,
		window_size,
		step,
		normalize=False,
		random_rot=False
	):
		"""
		Parameters:
		data_dir (str): path to the dataset directory
		window_size (int): sliding window size
		step (int): stride for sliding window
		normalize (bool): to normalize data using train split mean and variance
		"""
		self.data_dir = data_dir
		self.window_size = window_size
		self.step = step
		self.normalize = normalize
		self.random_rot = random_rot

		self.dataset_name = info['dataset']
		if normalize:
			mean = torch.tensor(info['mean'], dtype=torch.float64)
			std = torch.tensor(info['std'], dtype=torch.float64)
			self.mean = mean.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords
			self.std = std.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords

		self.label_map = info['label_map']
		self.joints_connections = info['joints_connections']
		self.n_joints = info['n_joints']
		self.non_gesture_idx = self.label_map.index("NON-GESTURE")

		self.load_data()

	def load_data(self):

		self.sequences = []
		self.labels = []
		self.labels_window = []

		with open(opt.join(self.data_dir, "annotations.txt"), "r") as annotations:
			## annotations should follow this format
			# sequence_Id;gesture_1;start_frame;end_frame;...;gesture_n;start_frame;end_frame;"
			# ...
			pbar = tqdm(annotations.readlines(), desc="Loading Online Training {} Dataset....".format(self.dataset_name), colour='green')
			for line in pbar:
				line = line.split(";")
				file_name = line[0]
				data = line[1:-1]

				file = str(file_name) + ".txt"
				with open(opt.join(self.data_dir, file), "r") as fd:
					n_frames = len(fd.readlines())
				gt_window = np.zeros(n_frames) + self.non_gesture_idx
				
				## scrap gesture for each sequence
				for index in range(0, len(data), 3):
					s = int(data[index + 1])
					e = int(data[index + 2])
					lab = data[index]
					gt_window[s : e + 1] = self.label_map.index(lab)

				sequence_data = []

				with open(opt.join(self.data_dir, f"{file_name}.txt"), "r") as fd:
					for line_idx, line_ in enumerate(fd.readlines()):
						line_ = line_.split(";")[1:-1]  # remove index and end-of-line
						line_ = np.reshape(line_, (self.n_joints, 3)).astype(np.float64)  
						sequence_data.append(line_)

				sequence_data = np.array(sequence_data).astype(np.float64)
				# generate windows of size W for current gesture
				for cursor in range(0, sequence_data.shape[0] - self.window_size, self.step):
					label_window = gt_window[cursor : cursor + self.window_size]
					## count most frequent label in that occurs in the window
					label_count = list(np.bincount(label_window.astype("int64")))
					if len(label_count) == 0:
						continue

					## assign that label to the window
					dom_label = label_count.index(max(label_count))
					if (label_count[-1] / len(label_window)) < 0.7:
						dom_label = label_count.index(max(label_count[:-1]))
						
					self.sequences.append(sequence_data[cursor : cursor + self.window_size, :, :])
					self.labels.append(dom_label)
					self.labels_window.append(label_window)
					
	def __len__(self):
		return len(self.sequences)

	def _is_gesture(self, label):
		return label != self.non_gesture_idx

	# def _type_gesture(self, label):
	# 	return self.gestures_type_map[label]

	def _normalize(self, x):
		'''
		normalize data with train mean and variance 
		'''
		return (x - self.mean) / self.std

	def _preprocess(self, sequence: np.ndarray):
		'''
		preprocessing the sequences
		'''
		sequence_tensor = torch.from_numpy(sequence)
		if self.normalize:
			sequence_tensor = self._normalize(sequence_tensor)
		return sequence_tensor.float()

	def __getitem__(self, item):
		sequence = self.sequences[item]
		sequence = self._preprocess(sequence)
		window_label = self.labels_window[item]
		single_label = self.labels[item]

		start_idx, end_idx = 0.0, 0.0
		is_start_valid = False
		is_end_valid = False
		lab = window_label[0]
		for idx, i in enumerate(window_label):
			if i != lab and lab == self.non_gesture_idx:
				is_start_valid = True
				start_idx = idx
			
			elif i == self.non_gesture_idx and lab != self.non_gesture_idx:
				is_end_valid = True
				end_idx = idx
			lab = i
			
		# normalize
		start_idx = start_idx / len(window_label)        
		end_idx = end_idx / len(window_label)        
		type_gesture = self._is_gesture(single_label)
		
		label  = torch.tensor(single_label).long()
		type_gesture  = torch.tensor(type_gesture).float()
		start_idx = torch.tensor(start_idx).float()
		end_idx = torch.tensor(end_idx).float()

		return dict(Sequence=sequence, 
					Label=label,
					Type_Gesture=type_gesture,
					Start_Idx=start_idx,
					End_Idx=end_idx,
					Is_Start_Valid=is_start_valid,
					Is_End_Valid=is_end_valid,
					Window_Label=window_label
					)


class OnlineEvalDataset(Dataset):
	def __init__(
		self,
		data_dir,
		file_idx,
		window_size,
		step,
		info,
		normalize=False,
	):
		"""
		Parameters:
		data_dir (str): path to the dataset directory
		file_idx (int): sequence Id
		window_size (int): sliding window size
		step (int): stride for sliding window
		normalize (bool): to normalize data using train split mean and variance
		"""

		self.data_dir = data_dir
		self.window_size = window_size
		self.step = step
		self.normalize = normalize
		self.file_idx = file_idx

		self.dataset_name = info['dataset']
		if normalize:
			mean = torch.tensor(info['mean'], dtype=torch.float64)
			std = torch.tensor(info['std'], dtype=torch.float64)
			self.mean = mean.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords
			self.std = std.unsqueeze(0).unsqueeze(0)  # window_size, joints, coords

		self.label_map = info['label_map']
		self.joints_connections = info['joints_connections']
		self.n_joints = info['n_joints']
		self.non_gesture_idx = self.label_map.index("NON-GESTURE")

		self.load_data()
		

	def load_data(self):

		self.sequences = []
		self.labels = []
		self.labels_window = []

		with open(opt.join(self.data_dir, "annotations.txt"), "r") as annotations:
			## annotations should follow this format
			# sequence_Id;gesture_1;start_frame;end_frame;...;gesture_n;start_frame;end_frame;"
			# ...
			pbar = tqdm(annotations.readlines(), desc="Loading Online Evalaution {} Dataset....".format(self.dataset_name), colour='green')
			for line in pbar:
				line = line.split(";")
				file_name = line[0]

				if file_name == str(self.file_idx):
					
					file = str(self.file_idx) + ".txt"
					with open(opt.join(self.data_dir, file), "r") as fd:
						n_frames = len(fd.readlines())
					self.gt_window = np.zeros(n_frames) + self.non_gesture_idx
   
					data = line[1:-1]
					
					## scrap gesture for each sequence
					for index in range(0, len(data), 3):
						# label=lab, start=s, end=e
						s = int(data[index + 1])
						e = int(data[index + 2])
						lab = data[index]
						if e >= n_frames:
							e = n_frames-1
						self.gt_window[s : e + 1] = self.label_map.index(lab)

					sequence_data = []

					with open(opt.join(self.data_dir, f"{file_name}.txt"), "r") as fd:
						for line_idx, line_ in enumerate(fd.readlines()):
							line_ = line_.split(";")[1:-1]  # remove index and end-of-line
							line_ = np.reshape(line_, (self.n_joints, 3)).astype(np.float64)  
							sequence_data.append(line_)

					sequence_data = np.array(sequence_data).astype(np.float64)
					# generate windows of size W for current gesture
					for cursor in range(0, sequence_data.shape[0] - self.window_size, self.step):
						label_window = self.gt_window[cursor : cursor + self.window_size]
						## count most frequent label in that occurs in the window
						label_count = list(np.bincount(label_window.astype("int64")))
						
						## assign that label to the window
						dom_label = label_count.index(max(label_count))

						self.sequences.append(sequence_data[cursor : cursor + self.window_size, :, :])
						self.labels.append(dom_label)
						self.labels_window.append(label_window)

		self.sequence_data = self._preprocess(sequence_data).float()
		self.gt_window_gesture = np.array([self._is_gesture(l) for l in self.gt_window])
		
	def __len__(self):
		return len(self.sequences)

	def _is_gesture(self, label):
		return label != self.non_gesture_idx

	def _normalize(self, x):
		'''
		normalize data with train mean and variance 
		'''
		return (x - self.mean) / self.std

	def _preprocess(self, sequence: np.ndarray):
		'''
		preprocessing the sequences
		'''
		sequence_tensor = torch.from_numpy(sequence)
		if self.normalize:
			sequence_tensor = self._normalize(sequence_tensor)
		return sequence_tensor.float()

	def __getitem__(self, index):
		sequence = self.sequences[index]
		sequence = self._preprocess(sequence)
		return dict(Sequence=sequence)
	


if __name__ == '__main__':

	## Pre-TRAINING
	data_dir = '/home/omar/Documents/Data/HGR datasets/IPN_HAND/test_set/'
	# data_set = PreTrainingDataset(data_dir,
	# 							window_size=16,
	# 							step=1,
	# 							normalize=False,
	# 							random_rot=False)

	# loader = DataLoader(data_set, batch_size=4, shuffle=True)

	# d = next(iter(loader))

	# print('Pretraining Training:')
	# print('> Window shape:', d['Sequence'].shape)
	# print()

	## OFFLINE TRAINING
	data_set = OfflineDataset(data_dir,
							sequence_length=80,
							normalize=False)

	loader = DataLoader(data_set, batch_size=4, shuffle=True)

	for d in loader:
		pass

	# d = next(iter(loader))

	# print('Offline Training:')
	# print('> Sequence shape :', d['Sequence'].shape)
	# print('> Label          :', d['Label'])
	# print()

	# ## ONLINE TRAINING
	# data_set = OnlineTrainingDataset(data_dir=data_dir,
	# 								 window_size=16,
	# 								 step=1,
	# 								 normalize=False)

	# loader = DataLoader(data_set, batch_size=4, shuffle=True)

	# d = next(iter(loader))

	# print('Online Training:')
	# print('> Window shape      :', d['Sequence'].shape)
	# print('> Label             :', d['Label'])
	# print('> Type_Gesture      :', d['Type_Gesture'])
	# print('> Start_Idx         :',  d['Start_Idx'])
	# print('> Is_Start_Valid    :', d['Is_Start_Valid'])
	# print('> End_Idx           :', d['End_Idx'])
	# print('> Is_End_Valid      :', d['Is_End_Valid'])
	# print('> Window_Label shape:', d['Window_Label'].shape)
	# print()

	# ## EVALUATION
	# data_set = OnlineEvalDataset(data_dir=data_dir,
	# 							 file_idx=1,
	# 							 window_size=16,
	# 							 step=1,
	# 							 normalize=False)

	# loader = DataLoader(data_set, batch_size=4, shuffle=True)

	# d = next(iter(loader))

	# print('Online Evalaution:')
	# print('> Window shape   :', d['Sequence'].shape)
	# print('> Sequence length:', data_set.gt_window.shape)