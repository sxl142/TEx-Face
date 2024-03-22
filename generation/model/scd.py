from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
from model.CLIP_encoder import CLIPTextEmbedding
import clip
from functools import partial
import math

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn
from model.rotary_embedding import RotaryEmbedding

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# absolute positional embedding used for vanilla transformer sequential data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

class Learnable_positional_encoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, 14, d_model))
    def forward(self, x):
        return self.dropout(x + self.pe)

# very similar positional embedding used for diffusion timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# dropout mask
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn_exp = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.layer_selection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softmax(dim=1)
        )
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory_txt,
        memory_exp,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # import pdb; pdb.set_trace()
        
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block_exp_txt(
                self.norm2(x), memory_txt, memory_exp, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block_exp_txt(x, memory_txt, memory_exp, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        # print(atten)
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)
    
    def _mha_block_exp_txt(self, x, mem1, mem2, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k1 = self.rotary.rotate_queries_or_keys(mem1) if self.use_rotary else mem1
        k2 = self.rotary.rotate_queries_or_keys(mem2) if self.use_rotary else mem2
        x_txt = self.multihead_attn(
            q,
            k1,
            mem1,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

        x_exp = self.multihead_attn_exp(
            q,
            k2,
            mem2,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x_txt + x_exp

        return self.dropout2(x)
    

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond_txt, cond_exp, t):
        for layer in self.stack:
            x = layer(x, cond_txt, cond_exp, t)
        return x


class Model(nn.Module):
    def __init__(
        self, opt
    ) -> None:

        super().__init__()
        self.input_dim = opt.input_dim
        self.output_dim = opt.output_dim
        self.en_layers = opt.en_layers
        self.embed_dim_en = opt.embed_dim_en
        self.head = opt.head
        self.drop_rate = opt.drop_rate
        self.cond_drop_prob = opt.cond_drop_prob
        self.use_rotary = opt.use_rotary

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        
        if self.use_rotary:
            self.rotary = RotaryEmbedding(dim=self.embed_dim_en)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                self.embed_dim_en, self.drop_rate, batch_first=True
            )

        self.clip_model = CLIPTextEmbedding(normalize=False)

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.embed_dim_en // 2),  # learned?
            nn.Linear(self.embed_dim_en // 2, self.embed_dim_en*2),
            nn.Mish(),
        )
        self.to_time_emb = nn.Sequential(nn.Linear(self.embed_dim_en*2, self.embed_dim_en),)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(self.embed_dim_en*2, self.embed_dim_en*2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )
        # null embeddings for guidance dropout
        self.null_cond_txt = nn.Parameter(torch.randn(1, 1, self.embed_dim_en))
        self.null_cond_exp = nn.Parameter(torch.randn(1, 1, self.embed_dim_en))

        # input projection
        self.input_projection = nn.Linear(self.input_dim, self.embed_dim_en)
       
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(self.en_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    self.embed_dim_en,
                    self.head,
                    dim_feedforward=self.embed_dim_en * 4,
                    dropout=self.drop_rate,
                    activation=F.gelu,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.exp_proj = nn.Sequential(
            nn.Linear(50, self.embed_dim_en),
           
        )
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(self.embed_dim_en, self.output_dim)

    def forward_with_cond_scale(self, x, t, cond, cond_scale=1, pos=None, exp=None, rescaled_phi=0, **kwargs):
        unc = self.forward(x, t, cond, pos, exp, cond_drop_prob=1)
        if cond_scale == 0:
            return unc
        conditioned = self.forward(x, t, cond, pos, exp, cond_drop_prob=0)
        scaled_logits = unc + (conditioned - unc) * cond_scale
        return scaled_logits
        

    def forward(
        self, x, t, cond, pos, exp, cond_drop_prob=None
    ):
        batch_size, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)
        
        time_emb = self.time_mlp(t)
        t_emb = self.to_time_emb(time_emb)
        t_tokens = self.to_time_tokens(time_emb)

        if cond != None:
            txt_emb_drop = clip.tokenize([''] * batch_size, truncate=True).to(x.device)
            txt_emb = clip.tokenize(cond, truncate=True).to(x.device)

            txt_emb[txt_emb < 0] = 0 # some padded text token maybe negative, so set them to 0
            txt_emb_drop[txt_emb_drop < 0] = 0 # some padded text token maybe negative, so set them to 0

            keep_mask = prob_mask_like((batch_size, 1), 1 - cond_drop_prob, device='cuda').expand(batch_size, 77)
            
            cond_tokens_txt = torch.where(keep_mask, txt_emb, txt_emb_drop)
            cond_tokens_txt = self.clip_model.encode_text(cond_tokens_txt).detach().float()

        else:
            txt_emb = clip.tokenize(['']* batch_size, truncate=True).to(x.device)
            txt_emb[txt_emb < 0] = 0 # some padded text token maybe negative, so set them to 0
            cond_tokens_txt = self.clip_model.encode_text(txt_emb).detach().float()


        if exp != None:
            cond_tokens_exp = self.exp_proj(exp).unsqueeze(1)
            keep_mask_exp = prob_mask_like(batch_size, 1 - cond_drop_prob, device='cuda')
            keep_mask_embed_exp = rearrange(keep_mask_exp, "b -> b 1 1")
            null_cond_exp_emb = self.null_cond_exp.to(cond_tokens_exp.dtype)
            cond_tokens_exp = torch.where(keep_mask_embed_exp, cond_tokens_exp, null_cond_exp_emb)
        else:
            cond_tokens_exp = self.null_cond_exp.repeat(batch_size, 1, 1)
            
        x_with_time_tokens = torch.cat((x, t_tokens), dim=1)
        output = self.seqTransDecoder(x_with_time_tokens, cond_tokens_txt, cond_tokens_exp, t_emb)
        output = self.final_layer(output)

        return output[:, :14]