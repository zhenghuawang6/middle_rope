import os
import sys
import pdb
import math
import copy
import time 
import types
from typing import Optional, Tuple
import numpy as np 
from scipy.stats import entropy

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, StaticCache
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

def apply_rotary_pos_emb_single_scaling(x, cos, sin, position_ids):
    cos = cos[:,position_ids]  # [head, bs, seq_len, dim]
    sin = sin[:,position_ids]  # [head, bs, seq_len, dim]

    cos = cos.transpose(0, 1)  # [bs, head, seq_len, dim]
    sin = sin.transpose(0, 1)  # [bs, head, seq_len, dim]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class MsPoELlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, num_patterns=2, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_patterns = num_patterns

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_patterns = self.num_patterns
        self.max_seq_len_cached = seq_len
        #t是一个[num_heads，max_seq_len_cached]的矩阵，每一行就是一个位置的索引[0，max_seq_len_cached-1]
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_patterns,1)
        compress_ratio = torch.arange(num_patterns, device=device, dtype=self.inv_freq.dtype)
        #根据头的数量进行压缩比例的创建
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (compress_ratio / num_patterns)
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
            self.cos_cached[0,:seq_len].to(dtype=x.dtype),
            self.sin_cached[0,:seq_len].to(dtype=x.dtype),
            self.cos_cached[1,:seq_len].to(dtype=x.dtype),
            self.sin_cached[1,:seq_len].to(dtype=x.dtype),
        )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx, min_cratio, max_cratio):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.layer_idx = layer_idx
        self.min_cratio = min_cratio
        self.max_cratio = max_cratio

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
    
    def _init_rope(self):

        self.rotary_emb = MsPoELlamaRotaryEmbedding(
            self.head_dim,#维度
            min_cratio=self.min_cratio,#最小的压缩比
            max_cratio=self.max_cratio,#最大的压缩比
            max_position_embeddings=self.max_position_embeddings,#最大的长度
            base=self.rope_theta,#基础的base
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        is_first = (past_key_value.get_seq_length == 0)
        #处理相关的cache,放入新的，拿到所有的
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        #拿到旋转角度
        big_cos, big_sin, small_cos, small_sin = self.rotary_emb(value_states, key_states.shape[-2])
        #比如attentionmask 确定上下文以及query 1是上下文  0是query以及后续的东西
        signals = attention_mask.squeeze()
        

        
        #计算注意力分数
        if is_first:
            #确定边界点

             

        else:
           #






        
        #需要改动
        cos, sin = position_embeddings

        

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
