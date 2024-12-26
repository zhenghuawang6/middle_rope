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
        self._init_scales()

    #测试temperature 将开始的注意力分给中间
    def _init_rope(self):

        self.rotary_emb1 = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=1)
        self.rotary_emb2 = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=1)

    #根据头部对位置的敏感程度进行干扰，对位置敏感的，干扰的少一点，对位置不敏感的干扰的多一点
    def _init_scales(self):
        #temperature从大到小，大的tem对应异常值较小的
        self.tem_matric =  torch.cat((torch.linspace(1.5,1,10,dtype=torch.float32),torch.ones([22]))).to("cuda")
        self.tem_matric =  torch.cat((torch.ones([22]),torch.linspace(1,1.3,10,dtype=torch.float32))).to("cuda")


    def _calculate_outlier(self, attn_weights):
        average = attn_weights.mean(-1).unsqueeze(-1)
        #广播机制  取最后一个token的异常值
        outlier = (attn_weights > 3 * average).float().mean(-1)[:,:,-1]
        #原始元素在小到大的list中排第几个
        head_orders = outlier.argsort(dim=-1).argsort(dim=-1)
        return head_orders
    
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
        #进行扩充
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

        # offset = int(context_length * (1/self.big_scale - 1/self.small_scale))
        big_cos, big_sin = self.rotary_emb1(value_states, seq_len=kv_seq_len)
        # small_cos, small_sin = self.rotary_emb2(value_states, seq_len=kv_seq_len+offset)
        small_cos, small_sin = self.rotary_emb2(value_states, seq_len=kv_seq_len)

        # small_cos = small_cos[:,:,offset:]
        # small_sin = small_sin[:,:,offset:]

        #上下文的index
        big_query_states, big_key_states = apply_rotary_pos_emb(query_states, key_states, big_cos, big_sin, position_ids) # normal attention 
        #后面的index
        small_query_states, small_key_states = apply_rotary_pos_emb(query_states, key_states, small_cos, small_sin, position_ids) # normal attention 
        

        if past_key_value is not None:
            # reuse k, v, self_attention
            big_key_states = torch.cat([past_key_value[0], big_key_states], dim=2)
            small_key_states = torch.cat([past_key_value[1], small_key_states], dim=2)     # cache group_key_states
            value_states = torch.cat([past_key_value[2], value_states], dim=2)

        if use_cache:
            past_key_value = (big_key_states, small_key_states, value_states) 
        else:
            past_key_value = None
        big_key_states = repeat_kv(big_key_states, self.num_key_value_groups)
        small_key_states = repeat_kv(small_key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        big_attn_weights = torch.matmul(big_query_states, big_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        small_attn_weights = torch.matmul(small_query_states, small_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 
        
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
            #无效了，因为两个注意力map都是一样的
            context_length = q_len - 11
            neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=big_attn_weights.device)
            neighbor_attention_mask = torch.tril(neighbor_attention_mask)
            group_attention_mask =  torch.tril(torch.ones((q_len-context_length, kv_seq_len), device=big_attn_weights.device),diagonal=context_length)
            neighbor_attention_mask[context_length:, :] -= group_attention_mask
        else:
            raise ValueError("q_len should be 1 or seq_len.")

        merged_attn_weights1 = torch.where(neighbor_attention_mask.bool(), big_attn_weights, small_attn_weights) # replace the group attention with neighbor attention within the neighbor window. 
        #从小到大的index，越小代表位置越不敏感，需要进行高温度的平滑
        merged_attn_weights = nn.functional.softmax(merged_attn_weights1, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        #寻找异常值
        head_orders = self._calculate_outlier(merged_attn_weights)
        #根据位置重新进行索引
        tem_matric = self.tem_matric[head_orders].unsqueeze(-1).unsqueeze(-1).to(hidden_states.device)
        #按照temperature进行缩放
        merged_attn_weights1 /= tem_matric
        #重新进行原来值的计算
        merged_attn_weights1 = torch.where(attention_mask.bool(), big_attn_weights, merged_attn_weights1) # replace the group attention with neighbor attention within the neighbor window. 
        merged_attn_weights = nn.functional.softmax(merged_attn_weights1, dim=-1, dtype=torch.float32).to(query_states.dtype) 

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
    
