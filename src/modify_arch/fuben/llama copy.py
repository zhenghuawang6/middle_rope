from typing import Optional, Tuple

import os
import sys
import pdb
import math
import copy
import time 
import types
import numpy as np 
from scipy.stats import entropy

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaLinearScalingRotaryEmbedding,
    LlamaForCausalLM,
)


__all__ = ['MsPoELlamaForCausalLM']


def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb_single_scaling(x, cos, sin, position_ids):
    cos = cos[:,position_ids]  # [head, bs, seq_len, dim]
    sin = sin[:,position_ids]  # [head, bs, seq_len, dim]

    cos = cos.transpose(0, 1)  # [bs, head, seq_len, dim]
    sin = sin.transpose(0, 1)  # [bs, head, seq_len, dim]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def sample_rotary_emb(cos, sin, num_key_value_groups):
    cos = cos[::num_key_value_groups,...]  # [head, bs, seq_len, dim]
    sin = sin[::num_key_value_groups,...]  # [head, bs, seq_len, dim]
    return cos, sin


### Positional Scaling
class MsPoELlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, num_heads=32, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        self.max_seq_len_cached = seq_len
        #t是一个[num_heads，max_seq_len_cached]的矩阵，每一行就是一个位置的索引[0，max_seq_len_cached-1]
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_heads,1)
        compress_ratio = torch.arange(num_heads, device=device, dtype=self.inv_freq.dtype)
        #根据头的数量进行压缩比例的创建
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (compress_ratio / num_heads)
        compress_ratio = compress_ratio.unsqueeze(-1)#[num_heads, 1]
        #每一行就是压缩后的位置index  t-》[num_heads，max_seq_len_cached] inv_freq-》[1,dim/2]
        t = t / compress_ratio

        freqs = torch.einsum("ki,j->kij", t, self.inv_freq) #[num_heads，max_seq_len_cached，dim/2]
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1) ##[num_heads，max_seq_len_cached，dim]
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    #
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:,:seq_len].to(dtype=x.dtype),
            self.sin_cached[:,:seq_len].to(dtype=x.dtype),
        )


class MsPoELlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.compress_ratio_min = config.compress_ratio_min
        self.compress_ratio_max = config.compress_ratio_max

        self.enable_head_metrics = True
        self.head_type = config.head_type
        self.head_order = None

        self._init_rope()

    def _head_wise_statistics(self, query_states, key_states, q_len, kv_seq_len, bsz, attention_mask):

        query_states_new = query_states
        # batch, num_key_value_heads, slen, head_dim
        key_states_new = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states_new, key_states_new.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.squeeze(0)

        head_orders = self._calculate_outlier(attn_weights)

        return head_orders


    def _calculate_outlier(self, attn_weights):
        # attn_weights: [num_heads, q_len, kv_seq_len]
        #[1,b,h,ql.kl]
        average = attn_weights.mean(-1).unsqueeze(-1)
        outlier = - (attn_weights > 3 * average).float().mean(-1)[:,-1]
        head_orders = outlier.argsort()

        if self.head_type == "normal":
            head_orders = np.arange(self.num_heads)
            head_orders = self.num_heads - head_orders - 1

        return head_orders


    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MsPoELlamaRotaryEmbedding(
                self.head_dim,#维度
                min_cratio=self.compress_ratio_min,#最小的压缩比
                max_cratio=self.compress_ratio_max,#最大的压缩比
                num_heads=self.num_heads,#头的数量
                max_position_embeddings=self.max_position_embeddings,#最大的长度
                base=self.rope_theta,#基础的base
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                assert False # not implemented
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    min_cratio=self.compress_ratio_min,
                    max_cratio=self.compress_ratio_max,
                    num_heads=self.num_heads,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                assert False # not implemented
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        #假如该attention mask是区分关于那些是context那些是query的标志
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        #输出的维度为[kvcache+tarlen,tarlen]
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        #统计tensor中元素的个数
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1
        #[num_heads，max_seq_len_cached，dim]
        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        if self.enable_head_metrics:
            self.head_order = self._head_wise_statistics(query_states, key_states, q_len, kv_seq_len, bsz, attention_mask)
            self.enable_head_metrics = False

        cos = cos[self.head_order, :, :]
        sin = sin[self.head_order, :, :]

        query_states = apply_rotary_pos_emb_single_scaling(query_states, cos, sin, position_ids)

        cos, sin = sample_rotary_emb(cos, sin, self.num_key_value_groups)
        key_states = apply_rotary_pos_emb_single_scaling(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # key/value are already rotated
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MsPoELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                self.model.layers[layer_idx].self_attn = MsPoELlamaAttention(config)

    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None

