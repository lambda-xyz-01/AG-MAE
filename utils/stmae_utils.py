import torch
from tqdm import tqdm
import os.path as opt
import os 

from utils.visualize import display_sample


def train_epoch(epoch, num_epochs, model, optimizer, dataloader, device, scheduler=None):
    model.train()
    train_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f'[%.3g/%.3g]' % (epoch, num_epochs), colour='green')
    for d in pbar:
        seq = d['Sequence'].to(device).float()
        optimizer.zero_grad()               
        *_, loss = model(seq)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(train_loss=f'{train_loss:.2f}')
        
    if scheduler is not None:
        scheduler.step()
    
    return train_loss

def valid_epoch(model, dataloader, device):
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc='[VALID]', colour='green')
        for d in pbar:
            seq = d['Sequence'].to(device).float()         
            *_, loss = model(seq)
            valid_loss += loss.item()
            pbar.set_postfix(valid_loss=f'{valid_loss:.2f}')

    return valid_loss


def stmae_training_loop(model, train_loader, valid_loader, device, optimizer, scheduler, args):

    save_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name,'weights/')
    img_save_folder_path = opt.join(args.save_folder_path, args.dataset, args.exp_name, 'reconstucted/')
    os.makedirs(img_save_folder_path, exist_ok=True)
    os.makedirs(save_folder_path, exist_ok=True)
    
    ## TRAINING
    print('\nTRAINING....')
    start_epoch = 1
    best_val = float("inf")

    ## training loop
    for epoch in range(start_epoch, args.stmae.num_epochs + 1):
        train_loss = train_epoch(epoch, args.stmae.num_epochs, model, optimizer, train_loader, device, scheduler)
        valid_loss = valid_epoch(model, valid_loader, device)
        
        is_best = valid_loss < best_val
        best_val = min(valid_loss, best_val)
        
        if is_best:
            torch.save(
                {'state_dict': model.state_dict(),
                 'best_val': best_val,
                 'epoch': epoch
                },
                opt.join(save_folder_path, "best_stmae_model.pth"),
            )

        out = next(iter(valid_loader))
        display_sample(model, out['Sequence'].to(device).float(), 
                        valid_loader.dataset.joints_connections,
                        fname=opt.join(img_save_folder_path, 'epoch_{}.png'.format(epoch)),
                        show=False)