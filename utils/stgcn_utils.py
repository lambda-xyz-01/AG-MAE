import math
import numpy as np
import matplotlib.pyplot as plt
import os.path as opt
import os 

import torch
from torch.nn import functional as F
from einops import rearrange, repeat
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_one_epoch(epoch, num_epochs, model, stmae, dataloader, optimizer, criterion, device, scheduler=None):
	model.train()
	stmae.eval()
	train_loss = 0

	pbar = tqdm(dataloader, total=len(dataloader), desc=f'[%.3g/%.3g]' % (epoch, num_epochs), colour='green')
	
	for d in pbar:
		seq = d['Sequence'].to(device).contiguous().float()
		label = d['Label'].to(device)            

		encoded_seq = stmae.inference(seq)
		
		## 
		optimizer.zero_grad()
		out = model(encoded_seq)
		
		## loss
		loss = criterion(out, label)              
		loss.backward()
		optimizer.step()
		
		train_loss += loss.item()
		pbar.set_postfix(train_loss=f'{train_loss:.2f}')

		break
		
	if scheduler is not None:
		scheduler.step()
	
	return train_loss
		
def valid_one_epoch(model, stmae, dataloader, criterion, device):
	accuracy, n, valid_loss = 0.0, 0, 0.0
	pred_labels, true_labels = [], []
	
	model.eval()
	stmae.eval()
	with torch.no_grad():
		pbar = tqdm(dataloader, total=len(dataloader), desc='[VALID]', colour='green')
		
		for d in pbar:
			seq = d['Sequence'].to(device).contiguous().float()
			label = d['Label'].to(device)
		
			encoded_seq = stmae.inference(seq)
			
			## predictions
			out = model(encoded_seq)

			## calc accuracy
			pred = out.argmax(dim=-1)
		
			accuracy += (pred == label.flatten()).sum().item()
			n += len(label)            
			pred_labels.extend(pred.tolist())
			true_labels.extend(label.tolist())
			
			## loss
			loss = criterion(out, label)
			valid_loss += loss.item()
			
			pbar.set_postfix(valid_loss=f'{valid_loss:.2f}', accuracy=f'{(accuracy / n)*100:.2f}%')

			break
				
	accuracy = (accuracy / n)*100
				
	return valid_loss, accuracy, true_labels, pred_labels


def train_one_epoch_online(epoch, num_epochs, model, stmae, dataloader, optimizer, criterion1, criterion2, device, scheduler=None):
	model.train()
	stmae.eval()

	total_loss = 0
	ng_idx = dataloader.dataset.non_gesture_idx
	pbar = tqdm(dataloader, total=len(dataloader), desc=f'[%.3g/%.3g]' % (epoch, num_epochs))
	
	for d in pbar:
		seq = d['Sequence'].to(device).contiguous().float()
		label = d['Label'].to(device)    
		type_gesture = d['Type_Gesture'].to(device)    
		start_idx = d['Start_Idx'].to(device)    
		end_idx = d['End_Idx'].to(device)    
		is_start_valid = d['Is_Start_Valid'].to(device)    
		is_end_valid = d['Is_End_Valid'].to(device)    
		
		## encode with STMAE encoder
		tokens = stmae.inference(seq)
		
		## classify with stgcn
		optimizer.zero_grad()
		out = model(tokens)
		class_proba, pred_type_gesture, pred_start, pred_end = out
		
		## loss
		loss = criterion1(class_proba, label) + criterion2(pred_type_gesture.squeeze(), type_gesture)
		if is_start_valid.any():
			loss += F.mse_loss(pred_start.squeeze()[is_start_valid], start_idx.squeeze()[is_start_valid])
		if is_end_valid.any():
			loss += F.mse_loss(pred_end.squeeze()[is_end_valid], end_idx.squeeze()[is_end_valid])

		pred_class = class_proba.argmax(dim=-1)
		
		loss.backward()
		optimizer.step()
		
		total_loss += loss.item()
		pbar.set_postfix(train_loss=f'{total_loss:.2f}')

	if scheduler is not None:
		scheduler.step()
	
	return total_loss
		

def valid_one_epoch_online(model, stmae, dataloader, criterion1, criterion2, device):
	accuracy, n, total_loss = 0.0, 0, 0.0
	pred_labels, true_labels = [], []
	
	model.eval()
	stmae.eval()

	ng_idx = dataloader.dataset.non_gesture_idx
	with torch.no_grad():
		pbar = tqdm(dataloader, total=len(dataloader), desc='[VALID]')
		
		for d in pbar:
			seq = d['Sequence'].to(device).contiguous().float()
			label = d['Label'].to(device)    
			type_gesture = d['Type_Gesture'].to(device)    
			start_idx = d['Start_Idx'].to(device)    
			end_idx = d['End_Idx'].to(device)    
			is_start_valid = d['Is_Start_Valid'].to(device)    
			is_end_valid = d['Is_End_Valid'].to(device)      
		
			tokens = stmae.inference(seq)
			
			## predictions
			out = model(tokens)
			class_proba, pred_type_gesture, pred_start, pred_end = out

			## loss
			loss = criterion1(class_proba, label) + criterion2(pred_type_gesture.squeeze(), type_gesture)
			if is_start_valid.any():
				loss += F.mse_loss(pred_start.squeeze()[is_start_valid], start_idx.squeeze()[is_start_valid])
			if is_end_valid.any():
				loss += F.mse_loss(pred_end.squeeze()[is_end_valid], end_idx.squeeze()[is_end_valid])
		
			total_loss += loss.item()
			
			pred_class = torch.zeros_like(type_gesture)
			pred_class[pred_type_gesture.squeeze() > 0.5] = 1.0

			pred_labels.extend(pred_class.tolist())
			true_labels.extend(type_gesture.tolist())
			
			accuracy += (pred_class == type_gesture).float().sum()
			n += len(type_gesture)
			
			pbar.set_postfix(valid_loss=f'{total_loss:.2f}', val_acc=f'{(accuracy/n)*100:.2f}%')

	accuracy = (accuracy / n) * 100
				
	return total_loss, accuracy, true_labels, pred_labels



def stgcn_online_training_loop(model, stmae, train_loader, valid_loader,  device, optimizer, criterion1, criterion2, scheduler, args):


	save_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name,'weights')
	save_thr_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name, 'confusion_matrix/')
	os.makedirs(save_folder_path, exist_ok=True)
	os.makedirs(save_thr_folder_path, exist_ok=True)

	model_args = args.stgcn_online

	## TRAINING
	print('\nTRAINING....')
	start_epoch = 1
	best_val = 0.0

	## training loop
	for epoch in range(start_epoch, model_args.num_epochs + 1): 
		train_loss = train_one_epoch_online(epoch, model_args.num_epochs,
											model, stmae, 
											train_loader,
											optimizer, 
											criterion1, criterion2, device, scheduler)

		valid_loss, accuracy, true_labels, pred_labels = valid_one_epoch_online(model, stmae, 
																				valid_loader, 
																				criterion1, criterion2, device)

		is_best = accuracy > best_val
		best_val = max(accuracy, best_val)
		
		if is_best:
			torch.save(
				{'state_dict': model.state_dict(),
				 'accuracy': accuracy,
				 'epoch': epoch
				},
				opt.join(save_folder_path, "best_online_stgcn_model.pth"),
			)


		fig, ax = plt.subplots(figsize=(10, 10))
		sns.heatmap(confusion_matrix(true_labels, pred_labels, normalize='true')*100,
					ax=ax, annot=True, fmt='.1f', 
					xticklabels=train_loader.dataset.label_map, yticklabels=train_loader.dataset.label_map)
		ax.set_title('Confusion Matrix (acc={})'.format(accuracy))
		plt.gcf().savefig(opt.join(save_thr_folder_path, 'cm_{}.png'.format(epoch)))


def stgcn_offline_training_loop(model, stmae, train_loader, valid_loader,  device, optimizer, criterion, scheduler, args):


	save_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name,'weights')
	save_thr_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name, 'confusion_matrix/')
	os.makedirs(save_folder_path, exist_ok=True)
	os.makedirs(save_thr_folder_path, exist_ok=True)

	model_args = args.stgcn_offline

	## TRAINING
	print('\nTRAINING....')
	start_epoch = 1
	best_val = 0.0

	## training loop
	for epoch in range(start_epoch, model_args.num_epochs + 1): 
		train_loss = train_one_epoch(epoch, model_args.num_epochs,
									model, stmae, 
									train_loader,
									optimizer, 
									criterion, device, scheduler)

		valid_loss, accuracy, true_labels, pred_labels = valid_one_epoch(model, stmae, 
																		valid_loader, 
																		criterion, device)

		is_best = accuracy > best_val
		best_val = max(accuracy, best_val)
		
		if is_best:
			torch.save(
				{'state_dict': model.state_dict(),
				 'accuracy': accuracy,
				 'epoch': epoch
				},
				opt.join(save_folder_path, "best_offline_stgcn_model.pth"),
			)


		fig, ax = plt.subplots(figsize=(10, 10))
		sns.heatmap(confusion_matrix(true_labels, pred_labels, normalize='true')*100,
					ax=ax, annot=True, fmt='.1f', 
					xticklabels=train_loader.dataset.label_map, yticklabels=train_loader.dataset.label_map)
		ax.set_title('Confusion Matrix (acc={})'.format(accuracy))
		plt.gcf().savefig(opt.join(save_thr_folder_path, 'cm_{}.png'.format(epoch)))


def get_best_detection_threshold(true_labels, pred_probas, fname=None):

	## get FPR & TPR at different threshold values
	fpr, tpr, thresholds = roc_curve(true_labels, pred_probas)
	roc_auc = auc(fpr, tpr)

	## calculate distances from each point on the ROC curve to the top-left corner
	distances = np.sqrt((1 - tpr)**2 + fpr**2)

	## find the index of the point with the minimum distance
	optimal_idx = np.argmin(distances)
	optimal_threshold = thresholds[optimal_idx]

	if fname is not None:
		## plot ROC curve
		plt.figure()
		plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic (ROC) Curve')
		plt.legend(loc="lower right")
		plt.gcf().savefig(fname)

	return optimal_threshold, roc_auc