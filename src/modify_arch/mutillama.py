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
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaForCausalLM,
)
class MutiLlamaAttention(nn.Module):
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
        self.is_causal = True
        self.narrow_scale = config.narrow_scale
        self.boost_scale = config.boost_scale
        self.context_len = config.context_len


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        #index间隔放大
        self.rotary_emb_boost = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.boost_scale)
        #index间隔减少
        self.rotary_emb_narrow = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.narrow_scale)
        # #index放大
        # self.rotary_emb1 = LlamaDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.big_scale)
        # #index减少
        # self.rotary_emb2 = LlamaDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.small_scale)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
       
        context_length = self.context_len
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        #也就是第一次，我需要给分配一个间隔的small - 20个tokne吧
        # if key_states.shape[-2] != -1:
        #     self.prefix_len = key_states.shape[-2]
        #进行位置
        boost_cos, boost_sin = self.rotary_emb_boost(value_states, seq_len=kv_seq_len)
        narrow_cos, narrow_sin = self.rotary_emb_narrow(value_states, seq_len=kv_seq_len+30)

        #将query与document相隔开来
        # small_cos = torch.cat((small_cos[:, :, :self.prefix_len-10, ...],small_cos[:, :, self.prefix_len+20:, ...]),dim=2)
        # small_sin = torch.cat((small_sin[:, :, :self.prefix_len-10, ...],small_sin[:, :, self.prefix_len+20:, ...]),dim=2)
        # small_cos = torch.cat((small_cos[:, :, :-19, ...],small_cos[:, :, -2:-1, ...]),dim=2)
        # small_sin = torch.cat((small_sin[:, :, :-, ...],small_sin[:, :, -2:-1, ...]),dim=2)

        
        #前面位置的
        boost_query_states, boost_key_states = apply_rotary_pos_emb(query_states, key_states, boost_cos, boost_sin, position_ids) 
        #后面位置的
        narrow_query_states, narrow_key_states = apply_rotary_pos_emb(query_states, key_states, narrow_cos, narrow_sin, position_ids) 
        
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            boost_key_states = torch.cat([past_key_value[0], boost_key_states], dim=2)
            narrow_key_states = torch.cat([past_key_value[1], narrow_key_states], dim=2) 
            value_states = torch.cat([past_key_value[2], value_states], dim=2)

        if use_cache:
            past_key_value = (boost_key_states, narrow_key_states, value_states) 
        else:
            past_key_value = None

        boost_key_states = repeat_kv(boost_key_states, self.num_key_value_groups)
        narrow_key_states = repeat_kv(narrow_key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        big_attn_weights = torch.matmul(boost_query_states, boost_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        small_attn_weights = torch.matmul(narrow_query_states, narrow_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            big_attn_weights = big_attn_weights + attention_mask
            small_attn_weights = small_attn_weights + attention_mask # causal mask. 
        

        if q_len == 1:
            # take effect with KV cache. 
            neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=big_attn_weights.device)
        elif q_len == kv_seq_len:
            #假设只有后十个token是narrow的
            context_length = q_len-10
            # offset = int(context_length * (1/self.big_scale - 1/self.small_scale))
            # no cache OR prefill
            neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=big_attn_weights.device)
            neighbor_attention_mask = torch.tril(neighbor_attention_mask)
            group_attention_mask =  torch.tril(torch.ones((q_len-context_length, kv_seq_len), device=big_attn_weights.device),diagonal=context_length)
            neighbor_attention_mask[context_length:, :] -= group_attention_mask
        else:
            raise ValueError("q_len should be 1 or seq_len.")

        merged_attn_weights = torch.where(neighbor_attention_mask.bool(), big_attn_weights, small_attn_weights) # replace the group attention with neighbor attention within the neighbor window. 
        merged_attn_weights = nn.functional.softmax(merged_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 


        attn_output = torch.matmul(merged_attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value