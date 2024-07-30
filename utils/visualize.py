
import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt
 
def draw_skeleton_data(skeleton_data, skeleton_connections, plot_connections=True):

    ## Plot the skeleton connections
    for connection in skeleton_connections:
        src = skeleton_data[connection[0]]
        dst = skeleton_data[connection[1]]
        if plot_connections:
                plt.plot([src[0], dst[0]], [src[1], dst[1]], marker='o')
        plt.scatter([src[0]], [src[1]], marker='o')
    plt.axis('off')
    
    
def display_sample(model, sequence, connections, fname=None, show=True):    
    
    window_size = sequence.shape[1]
    model.eval()

    ## reconstruct missing joints
    preds, masked_patches, masked_indices, loss = model(sequence)
    
    ## 
    batch_range = torch.arange(masked_patches.shape[0], device=sequence.device)[:, None]
    original_patches = sequence.detach().cpu()
    original_patches = rearrange(original_patches, 'b f n d -> b (f n) d')
    unmasked_nodes = original_patches.clone()
    
    recon_patches = original_patches.clone()
    recon_patches[batch_range, masked_indices.detach().cpu()] = preds.detach().cpu()
    unmasked_nodes[batch_range, masked_indices.detach().cpu()] = torch.nan
    
    original_patches = rearrange(original_patches, 'b (f n) d -> b f n d', f=window_size)
    recon_patches = rearrange(recon_patches, 'b (f n) d -> b f n d', f=window_size)
    unmasked_nodes = rearrange(unmasked_nodes, 'b (f n) d -> b f n d', f=window_size)

    plt.figure(figsize=(25, 5))
    for i in range(window_size):
        plt.subplot(3, window_size, i + 1)
        draw_skeleton_data(sequence[0][i].detach().cpu().numpy(), connections)
        plt.title('Frame {}'.format(i+1))

        plt.subplot(3, window_size, window_size + i + 1)
        draw_skeleton_data(unmasked_nodes[0][i].detach().cpu().numpy(), connections, plot_connections=True)

        plt.subplot(3, window_size, 2 * window_size + i + 1)
        draw_skeleton_data(recon_patches[0][i].detach().cpu().numpy(), connections)

    plt.suptitle('Top: Original Sequence | Middle: Masked Sequence | Bottom: Reconstructed Sequence')
    plt.gcf().tight_layout()
    if fname is not None:
        plt.gcf().savefig(fname)
    if show:
        plt.show()