import torch
import os 
import sys
import scipy
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
# from timm.layers import PatchEmbed
from timm.models.layers import PatchEmbed
import pdb


class Rearrange_Tensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Rearranges input tensor from shape (batch, channels, height, width, frames)
        to shape (batch, number of patches, patch dimension).
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch, channels, height, width, frames).
        - patch_size (tuple): Patch size for height, width, and frames dimensions (default: (2, 2, 2)).
        
        Returns:
        - torch.Tensor: Output tensor of shape (batch, number of patches, patch dimension).
        """
        # Unpack patch dimensions for height, width, and frames
        ph, pw, pf = (16, 16, 16)
        
        # Make sure height, width, and frames are divisible by the patch size
        assert x.shape[2] % ph == 0 and x.shape[3] % pw == 0 and x.shape[4] % pf == 0, \
            "Height, width, and frames must be divisible by the patch size."
        
        # Rearrange the tensor into patches
        # (batch, channels, height, width, frames) -> (batch, channels, patches_h, ph, patches_w, pw, patches_f, pf)
        x = rearrange(x, 'b c (ph patch_h) (pw patch_w) (pf patch_f) -> b (patch_h patch_w patch_f) (c ph pw pf)', 
                    ph=ph, pw=pw, pf=pf)
        
        # Output shape: (batch, number of patches, patch dimension)
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class PositionEmbeddingLearned3d(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256,h_patch_num = 16, w_patch_num = 16,d_patch_num = 64):
        super().__init__()
        self.h_patch_num = h_patch_num
        self.w_patch_num = w_patch_num
        self.d_patch_num = d_patch_num
        # print('num_pos_feats_shape:', num_pos_feats)
        
        self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        self.col_embed = nn.Embedding(w_patch_num, num_pos_feats)
        self.dep_embed = nn.Embedding(d_patch_num, num_pos_feats)
        # self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        # self.col_embed = nn.Embedding(w_patch_num, num_pos_feats+1)
        # self.dep_embed = nn.Embedding(d_patch_num, num_pos_feats+1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(self, B, h, w, d,x):
        i = (torch.arange(h, device=x.device) + 1)* (self.h_patch_num // h) -1
        j = (torch.arange(w, device=x.device) + 1)* (self.w_patch_num // w) -1
        k = (torch.arange(d, device=x.device) + 1)* (self.d_patch_num // d) -1
        x_emb = self.row_embed(i).unsqueeze(1).unsqueeze(2).repeat(1,w,d,1) #[8,16,16,256]
        y_emb = self.col_embed(j).unsqueeze(0).unsqueeze(2).repeat(h,1,d,1) #[16,8,16,256]
        z_emb = self.dep_embed(k).unsqueeze(0).unsqueeze(1).repeat(h,w,1,1) #[16,16,8,256]
        # print('x,y,z embeded: ', x_emb.shape, y_emb.shape, z_emb.shape)
        pos = torch.cat([x_emb,y_emb,z_emb,], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        pos = rearrange(pos,'b h w d c -> b (h w d) c') 
        return pos

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ViT (Vision Transformer) class definition
class ViT(nn.Module):
    def __init__(self, *, img3d_size, img3d_frame, patch3d_size, 
                patch3d_frame, dim, depth, heads,
                mlp_dim, pool = 'cls', channels = 4, dim_head = 64, 
                dropout = 0., emb_dropout = 0.):
        super().__init__()

        # Initialize image dimensions and patch sizes
        self.image3d_height, self.image3d_width = pair(img3d_size) # img3d_size: 256
        self.frames = img3d_frame # img3d_frame: 128
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.patch_height, self.patch_width = pair(patch3d_size) # patch3d_size: 16

        # Ensure image dimensions and frame sizes are divisible by patch sizes
        assert self.image3d_height % self.patch_height == 0 and self.image3d_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert img3d_frame % patch3d_frame == 0, 'Frames must be divisible by frame patch size'

        self.patch_height = self.patch_height
        self.patch_width = self.patch_width
        self.frame_patch_size = patch3d_frame

        # Calculate the number of patches and patch dimension
        # num_pathces 128,128,128->8*8*8=512
        num_patches = (self.image3d_height // self.patch_height) * (self.image3d_width // self.patch_width) * (self.frames // self.frame_patch_size) # 16*16*16=2048 check
        self.patch_dim = channels * self.patch_height * self.patch_width * self.frame_patch_size #16*16*16=4096
                                                                                                                
        # Ensure pooling type is valid
        assert pool in {'cls', 'mean', 'none'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        # Define patch embedding layer
        self.to_patch_embedding = nn.Sequential(
            # Rearrange the input tensor from shape (batch, channels, height, width, frames) 
            # to shape (batch, number of patches, patch dimension)
            # Rearrange('b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)', # (h p1): 高度方向上的分块 (w p2): 宽度方向上的分块 (f pf): 深度方向上的分块
            #             p1 = self.patch_height, 
            #             p2 = self.patch_width, 
            #             pf = self.frame_patch_size),
            Rearrange_Tensor(),
            # Apply Layer Normalization to the patch dimension
            nn.LayerNorm(self.patch_dim), 
            # Apply a Linear transformation to map the patch dimension to the model dimension
            nn.Linear(self.patch_dim, self.dim),
            # Apply Layer Normalization to the model dimension
            nn.LayerNorm(self.dim),
        )

        # Define positional embedding and transformer layers
        self.pos_embedding = PositionEmbeddingLearned3d(dim // 3,
                            (self.image3d_height // self.patch_height), 
                            (self.image3d_width // self.patch_width), 
                            (self.frames // self.frame_patch_size))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pos_drop = nn.Dropout(p=0)



    def _pos_embed(self, x):
        x = torch.cat((self.cls_token_2d.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed_2d
        return self.pos_drop(x)             


    def forward_3d(self, x_3d):
        B, C, H, W, D = x_3d.shape # (2,1,256,256,128)
        x_3d = self.to_patch_embedding(x_3d).contiguous() # (b, patch_num, feature_dim)
        # Print shapes to debug
        # print(f"x_3d shape after patch embedding: {x_3d.shape}")
        self.pos = self.pos_embedding(B, 
                                H // self.patch_height, 
                                W // self.patch_width, 
                                D // self.frame_patch_size,
                                x_3d
                                )
         # Print shape of positional embeddings
        # print(f"Positional embedding shape: {self.pos.shape}")
        x_3d += self.pos
        x_3d = self.dropout(x_3d)
        
        return x_3d

    def forward(self, x_3d):
        x_3d = self.forward_3d(x_3d) # # 图像变成patch featrues [b, 512, 768]

        x_3d = self.transformer(x_3d) # # 输入transformer,得到的结果维度和输入一样都是[b,512,768]
        # pdb.set_trace()
        if self.pool == 'none':
            pass
        elif self.pool == 'mean':            
            x_3d = x_3d.mean(dim = 1) # [2, 768]
        else: # 这里原来没有，bug在这里
            x_3d = x_3d[:, 0, :]  # Selects the first token (CLS token), shape [batch_size, hidden_dim]
        return x_3d

def vit_tiny(img3d_size, img3d_frame, patch3d_size, 
        patch3d_frame,
        dim = 192,
        depth = 12,
        heads = 3,
        mlp_dim = 768,
        dropout = 0.1,
        emb_dropout = 0.1,
        **kwargs):
    model_args = dict(img3d_size=img3d_size, 
                      img3d_frame=img3d_frame, 
                      patch3d_size=patch3d_size, 
                        patch3d_frame=patch3d_frame,
                        dim=dim,
                        depth=depth,
                        heads=heads,
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        emb_dropout=emb_dropout,
                        **kwargs)
    return ViT(**model_args)


def vit_small(img3d_size, img3d_frame, patch3d_size, 
        patch3d_frame,
        dim = 384,
        depth = 12,
        heads = 6,
        mlp_dim = 1536,
        dropout = 0.1,
        emb_dropout = 0.1,
        **kwargs):
    model_args = dict(img3d_size=img3d_size, 
                      img3d_frame=img3d_frame, 
                      patch3d_size=patch3d_size, 
                        patch3d_frame=patch3d_frame,
                        dim=dim,
                        depth=depth,
                        heads=heads,
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        emb_dropout=emb_dropout,
                        **kwargs)
    return ViT(**model_args)

# vit_middle的时候input_dim是512， 但是512//3无法整除，因此在position embedding concate
# 会三个方向的X,Y,Z的embedding后，会170*3=510所以会出现维度不匹配.所以我这里设置成513
def vit_middle(img3d_size, img3d_frame, patch3d_size, 
        patch3d_frame,
        dim = 513, #512
        depth = 12,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        **kwargs):
    model_args = dict(img3d_size=img3d_size, 
                      img3d_frame=img3d_frame, 
                      patch3d_size=patch3d_size, 
                        patch3d_frame=patch3d_frame,
                        dim=dim,
                        depth=depth,
                        heads=heads,
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        emb_dropout=emb_dropout,
                        **kwargs)
    return ViT(**model_args)


def vit_base(img3d_size, img3d_frame, patch3d_size, 
        patch3d_frame, 
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
        dropout = 0.1,
        emb_dropout = 0.1,
        **kwargs):
    model_args = dict(img3d_size=img3d_size, 
                      img3d_frame=img3d_frame, 
                      patch3d_size=patch3d_size, 
                        patch3d_frame=patch3d_frame,
                        dim=dim,
                        depth=depth,
                        heads=heads,
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        emb_dropout=emb_dropout,
                        **kwargs)
    return ViT(**model_args)


def test_vit():
    # Define input dimensions
    img3d_size_1 = 128
    img3d_frame_1 = 128
    img3d_size_2 = 224
    img3d_frame_2 = 224

    # Define patch sizes
    patch3d_size = 16
    patch3d_frame = 16

    # List of models to test
    models = {
        # "ViT Tiny": vit_tiny,
        # "ViT Small": vit_small,
        # "ViT Middle": vit_middle,
        "ViT Base": vit_base,
    }

    for model_name, model_fn in models.items():
        # Test the model with input size 128x128x128
        import pdb
        pdb.set_trace()
        print(f"\nTesting {model_name} with input size 128x128x128...")
        model_1 = model_fn(img3d_size=img3d_size_1, img3d_frame=img3d_frame_1,
                           patch3d_size=patch3d_size, patch3d_frame=patch3d_frame)
        x_3d_1 = torch.randn(2, 1, img3d_size_1, img3d_size_1, img3d_frame_1)  # Batch size 2
        output_1 = model_1(x_3d_1)
        print(f"Output shape: {output_1.shape}")

        # # Test the model with input size 224x224x224
        # print(f"\nTesting {model_name} with input size 224x224x224...")
        # model_2 = model_fn(img3d_size=img3d_size_2, img3d_frame=img3d_frame_2,
        #                    patch3d_size=patch3d_size, patch3d_frame=patch3d_frame)
        # x_3d_2 = torch.randn(2, 1, img3d_size_2, img3d_size_2, img3d_frame_2)  # Batch size 2
        # output_2 = model_2(x_3d_2)
        # print(f"Output shape: {output_2.shape}")

if __name__ == "__main__":
    test_vit()
