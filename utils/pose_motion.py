import torch
from torch.nn import functional as F

def poses_diff(x):
    _, H, W, _ = x.shape

    # x.shape (batch,channel,joint_num,joint_dim)
    x = x[:, 1:, ...] - x[:, :-1, ...]

    # x.shape (batch,joint_dim,channel,joint_num,)
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(H, W),
                    mode='nearest')
    x = x.permute(0, 2, 3, 1)
    # x.shape (batch,channel,joint_num,joint_dim)
    return x

def poses_motion(P):
    dist = pad_sequence((P[:, 1:, :, :] - P[:, :-1, :, :]), P.shape[1])
    P_diff = torch.flatten(dist, start_dim=2).abs().mean(dim=-1)
    return P_diff

def pad_sequence(seq, target_length):
    b, t, n, d = seq.shape
    if t >= target_length:
        return seq
    pad_len = target_length - t
    padding = torch.zeros((b, pad_len, n, d), dtype=seq.dtype).to(seq.device)
    return torch.cat([padding, seq], dim=1)