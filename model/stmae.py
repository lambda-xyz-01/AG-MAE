import torch
from torch import nn

from einops import rearrange

"""
    Cited and modified from
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


import torch
from torch import nn
import math
import numpy as np

"""
    Cited and modified from
    https://github.com/facebookresearch/3detr/blob/main/models/position_embedding.py
"""


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
                       ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
               ) + dst_range[0][:, None, :]
    return prop_xyz


class PosEmbFactory(nn.Module):
    def __init__(self,
                 temperature=10000,
                 normalize=False,
                 scale=None,
                 emb_type="fourier",
                 d_pos=None,
                 d_in=3,
                 gauss_scale=1.0, ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        assert emb_type in ["sin_v1", "sin_v2", "fourier"]
        self.emb_type = emb_type

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        if self.emb_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    @staticmethod
    def get_sinusoidal_emb_v1(self, n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def get_sinusoidal_emb_v2(self, xyz, n_channels, input_range):
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = n_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = n_channels - (ndim * xyz.shape[2])

        assert (
                ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={n_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_emb(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.emb_type == "sin_v1":
            with torch.no_grad():
                return self.get_sinusoidal_emb_v1(xyz, num_channels, input_range)
        elif self.emb_type == "sin_v2":
            with torch.no_grad():
                return self.get_sinusoidal_emb_v2(xyz, num_channels, input_range)
        elif self.emb_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_emb(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def __str__(self):
        st = f"type={self.emb_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st


    
import os
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import numpy as np

"""
    Cited and modified from
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mae.py
"""

lambda_anatomic = 1.0


class Encoder(nn.Module):
    def __init__(self, *,
                 patch_num,  # joint num
                 patch_dim,  # the number of sampling frame * joint coordinate shape
                 num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.,
                 window_size=15):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_dim = patch_dim
        self.to_patch_embedding = PosEmbFactory(emb_type="fourier", d_pos=dim)
        self.window_size = window_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, patch_num, dim))
        frame_idx = window_size
        joint_idx = patch_num // window_size
        pos_idx = torch.from_numpy(np.array([[(x, y) for x in range(frame_idx) for y in range(joint_idx)]])).to(
            torch.float32)
        pos_emb_factory = PosEmbFactory(emb_type="fourier", d_in=2, d_pos=dim)
        self.pos_embedding = pos_emb_factory(pos_idx)
        self.pos_embedding = self.pos_embedding.permute(0, 2, 1).cuda()

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )  # extract hidden space on the first row to get classification

    def forward(self, skel_data):
        x = self.to_patch_embedding(skel_data)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

###############
## STMAE 
###############

class STMAE(nn.Module):
    def __init__(self,
                 *,
                 encoder,
                 decoder_dim,
                 masking_strategy='random',
                 spatial_masking_ratio=0.5,
                 temporal_masking_ratio=0.5,
                 anatomical_loss=None,
                 decoder_depth=1,
                 decoder_heads=8,
                 decoder_dim_head=64):
        super().__init__()
        assert 0 < spatial_masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        assert 0 < temporal_masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.spatial_masking_ratio = spatial_masking_ratio
        self.temporal_masking_ratio = temporal_masking_ratio
        self.masking_strategy = masking_strategy
        self.window_size = encoder.window_size

        # encoder parameters
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.patch_to_emb = encoder.to_patch_embedding

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_prediction = nn.Linear(decoder_dim, self.encoder.patch_dim)
        
        
        self.anatomical_loss = anatomical_loss

    def load_checkpoint(self, load_dir, tag=None):
        load_path = os.path.join(
            load_dir,
            str(tag) + ".pth",
        )
        client_states = torch.load(load_path)
        state_dict = client_states['model']
        self.load_state_dict(state_dict, strict=True)
        return load_path, client_states
    
    def _generate_mask(self, P, strategy):

        b, T, N, _ = P.shape 
        device = P.device
                
        if strategy == 'random':

            num_masked = int(T * N * self.spatial_masking_ratio)

            ## generate a random noise
            noise = torch.rand(b, T * N, device=device)

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            masked_indices = ids_shuffle[:, :num_masked]
            unmasked_indices = ids_shuffle[:, num_masked:]

    
        elif strategy == 'temporal':
            ## number of frames to mask
            num_masked = int(T * self.temporal_masking_ratio)

            ## generate a random noise
            noise = torch.rand(b, T, device=device)

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            ## create a tensor representing the order of each frame
            masked_frames = ids_shuffle[:, :num_masked]
            masked_frames = masked_frames.view(b, -1, 1) * N

            unmasked_frames = ids_shuffle[:, num_masked:]
            unmasked_frames = unmasked_frames.view(b, -1, 1) * N

            ## use broadcasting to add frame_order to indices_first_frame
            frame_order = torch.arange(0, N, device=device)

            masked_indices = masked_frames + frame_order
            masked_indices = masked_indices.view(b, -1)

            unmasked_indices = unmasked_frames + frame_order
            unmasked_indices = unmasked_indices.view(b, -1)

            num_masked = N * num_masked
                
        elif strategy == 'spatial':  
            ## number of frames to mask
            num_masked = int(N * self.spatial_masking_ratio)

            ## generate a random noise
            noise = torch.rand(b, N, device=device)

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            frame_order = (torch.arange(0, T, device=device) * N).unsqueeze(1).T
            
            ## create a tensor representing the order of each frame
            masked_nodes = ids_shuffle[:, :num_masked].unsqueeze(-1)
            masked_indices = masked_nodes + frame_order
            masked_indices = masked_indices.view(b, -1)

            unmasked_nodes = ids_shuffle[:, num_masked:].unsqueeze(-1)
            unmasked_indices = unmasked_nodes + frame_order
            unmasked_indices = unmasked_indices.view(b, -1)
    
            num_masked = T * num_masked


        elif strategy == 'spatio-temporal':
            ## frame masking
            num_masked_frames = int(T * self.temporal_masking_ratio)
            noise = torch.rand(b, T, device=device)
            ids_shuffle = torch.argsort(noise, dim=1) 
            masked_frames = ids_shuffle[:, :num_masked_frames]
            unmasked_frames = ids_shuffle[:, num_masked_frames:]
            
            ## joints masking
            num_masked_joints = int(N * self.spatial_masking_ratio)
            noise = torch.rand(b, N, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)

            ## masked joints
            masked_frame_order = (masked_frames * N).unsqueeze(1)
            masked_frames_joints_1 = masked_frame_order + ids_shuffle.unsqueeze(-1)
            masked_frames_joints_1 = masked_frames_joints_1.view(b, -1)
            
            unmasked_frame_order = (unmasked_frames * N).unsqueeze(1)

            unmasked_frames_joints_idx = unmasked_frame_order + ids_shuffle.unsqueeze(-1)
            _, orgt, _ = unmasked_frames_joints_idx.shape
            unmasked_frames_joints_idx = rearrange(unmasked_frames_joints_idx, 'b n t -> b t n')
            
            b_, t_, n_ = unmasked_frames_joints_idx.shape
            noise_ = torch.rand(b_, t_, n_, device=device)
            ids_shuffle_ = torch.argsort(noise_, dim=-1)
            
            unmasked_frames_joints_idx = torch.gather(unmasked_frames_joints_idx, -1, ids_shuffle_)

    
            masked_frames_joints_2 = unmasked_frames_joints_idx[:, :, :num_masked_joints]
            masked_frames_joints_2 = rearrange(masked_frames_joints_2, 'b n t -> b (t n)')
            masked_indices = torch.cat((masked_frames_joints_1, masked_frames_joints_2), dim=-1)
        
            ## umasked joints
            unmasked_joints = unmasked_frames_joints_idx[:, :, num_masked_joints:]
            unmasked_indices = rearrange(unmasked_joints, 'b n t -> b (t n)')
            # unmasked_indices = unmasked_joints.view(b, -1)

            num_masked = (num_masked_frames  * N) + (T - num_masked_frames) * num_masked_joints
        
        elif strategy == 'motion-guided':
            
            num_masked = int(T * self.temporal_masking_ratio)
            
            avg_disp = poses_motion(P)
            max_val = avg_disp.max()
            avg_disp[:, 0] = max_val + 1.0
            avg_disp[:, -1] = max_val + 1.0
            
            values, indices = torch.topk(avg_disp, k=num_masked, dim=-1, largest=False, sorted=False)
            masked_indices, _ = torch.sort(indices)
                        
            masked_frames = masked_indices.view(b, -1, 1) * N
            ## use broadcasting to add frame_order to indices_first_frame
            frame_order = torch.arange(0, N, device=device)
            masked_indices = masked_frames + frame_order
            masked_indices = masked_indices.view(b, -1)
            
            # Create a tensor of all indices
            all_indices = torch.arange(N*T, device=device)
            unmasked_indices = []
            for row in masked_indices:
                unmasked_mask = ~torch.isin(all_indices, row)
                unmasked_idx = torch.nonzero(unmasked_mask).squeeze()
                unmasked_indices.append(unmasked_idx)
            unmasked_indices = torch.stack(unmasked_indices)

            num_masked = N * num_masked

        else:
            raise NotImplementedError
            
        return num_masked, masked_indices, unmasked_indices

    def forward(self, x):


        device = x.device
        b, f, n, _ = x.shape

        # #generate mask
        num_masked, masked_indices, unmasked_indices = self._generate_mask(x, self.masking_strategy)
        
        x = rearrange(x, 'b f n d -> b (f n) d')
        
        batch, num_patches, *_ = x.shape
        

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(x)
        tokens = tokens.permute(0, 2, 1)  # TO: batch, patch_size, hidden_dim
        tokens = tokens + self.encoder.pos_embedding

        # # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        # if force_rand_idx is not None:
        #     masked_indices = force_rand_idx[0]
        #     unmasked_indices = force_rand_idx[1]
        #     num_masked = masked_indices.shape[1]
        # else:
        #     num_masked = int(self.masking_ratio * num_patches)
        #     rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        #     masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        skel_masked = x[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(unmasked_tokens)

        # project encoder to decoder dimensions, if they are not equal
        # the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        skel_masked_pred = self.to_prediction(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(skel_masked_pred, skel_masked)
        if self.anatomical_loss is not None:
            rec_x = x.clone()
            rec_x[batch_range, masked_indices] = skel_masked_pred
            rec_x = rearrange(rec_x, 'b (f n) d -> b f n d', f=self.window_size)
            ang_loss, len_loss = self.anatomical_loss(rec_x)
            recon_loss = recon_loss + lambda_anatomic * (ang_loss + len_loss)
        return skel_masked_pred, skel_masked, masked_indices, recon_loss

    def change_ratio(self, ratio):
        self.masking_ratio = ratio
    
    def inference(self, x):
        b, *_ = x.shape
        x = rearrange(x, 'b (f w) n d -> (b f) w n d', w=self.window_size)
        x = rearrange(x, 'b w n d -> b (w n) d')
        with torch.no_grad():
            tokens = self.patch_to_emb(x)
            tokens = tokens.permute(0, 2, 1)
            tokens = tokens + self.encoder.pos_embedding
            encoded_tokens = self.encoder.transformer(tokens)
            encoded_tokens = self.enc_to_dec(encoded_tokens)
        encoded_tokens = rearrange(encoded_tokens, '(b f) (w n) d -> b (f w) n d', b=b, w=self.window_size)
        return encoded_tokens