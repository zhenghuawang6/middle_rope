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
from transformers.utils import logging
from transformers.cache_utils import Cache

logger = logging.get_logger(__name__)

class NewLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        #让index间隔变窄变宽的比例
        self.narrow_scale = config.narrow_scale
        self.boost_scale = config.boost_scale
        self.context_len = config.context_len


        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        #正常的旋转位置编码
        # self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self._init_rope()

    def _init_rope(self):
        #index间隔放大
        self.rotary_emb_boost = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.boost_scale)
        #index间隔减少
        self.rotary_emb_narrow = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.narrow_scale)
        #index间隔正常
        self.rotary_emb_norm = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=1.0)

        # #index放大
        # self.rotary_emb1 = LlamaDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.big_scale)
        # #index减少
        # self.rotary_emb2 = LlamaDynamicNTKScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=self.small_scale)
    #k[bs,hn,sq,dim]
    def apply_rotary_pos_emb_k(self, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return k_embed

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

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        position_embeddings = None
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            boost_cos, boost_sin = self.rotary_emb_boost(value_states, position_ids)
            narrow_cos, narrow_sin = self.rotary_emb_narrow(value_states, position_ids)
            
        else:
            cos, sin = position_embeddings

        #设置滑动窗口的位置编码
        boost_query_states, boost_key_states = apply_rotary_pos_emb(query_states, key_states, boost_cos, boost_sin)
        narrow_query_states, narrow_key_states = apply_rotary_pos_emb(query_states, key_states, narrow_cos, narrow_sin)

        #将两个合并，在第一个维度进行合并
        cat_key = torch.cat((boost_key_states, narrow_key_states), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            #update是从第二个维度进行合并
            cat_key, value_states = past_key_value.update(cat_key, value_states, self.layer_idx)

        kv_key_states = repeat_kv(cat_key, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        #进行切分
        boost_key_states, narrow_key_states = torch.chunk(kv_key_states, 2, dim=-1)

        #根据qk计算两个注意力矩阵
        boost_attn_weights = torch.matmul(boost_query_states, boost_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        narrow_attn_weights = torch.matmul(narrow_query_states, narrow_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        kv_seq_len = boost_key_states.shape[-2]
        first_token = 16
        recent_token = 100
        sliding_total = first_token + recent_token
        #根据一些条件进行合并
        if q_len == 1:
            location_attention_mask = torch.zeros((q_len, kv_seq_len), device=boost_attn_weights.device)
        elif q_len == kv_seq_len:
            #得到注意力分数[bs, sq, dim]
            norm_cos, norm_sin = self.rotary_emb_norm(value_states, position_ids)
            #拿出对应位置的position—id
            position_index = torch.argmax(position_ids == first_token-1, dim=-1)
            #一个batch一个batch的处理

            norm_cos = norm_cos[:,:sliding_total,...]
            norm_cos = norm_sin[:,:sliding_total,...]
            #需要设置sliding window注意力map
            for i in range(q_len):
                #进行旋转
                #如果q_len大于sliding_total，就需要往中间填充值
                if q_len >= sliding_total:
                    
                else:


                #对k进行旋转
                self.apply_rotary_pos_emb_k(k=kv_key_states,)

                #计算注意力分数

                #进行拼接

            context_length = q_len-10
            location_attention_mask = torch.ones((q_len, kv_seq_len), device=boost_attn_weights.device)
            location_attention_mask = torch.tril(location_attention_mask)
            query_attention_mask =  torch.tril(torch.ones((q_len-context_length, kv_seq_len), device=boost_attn_weights.device), diagonal=context_length)
            location_attention_mask[context_length:, :] -= query_attention_mask

        attn_weights = torch.where(location_attention_mask.bool(), boost_attn_weights, narrow_attn_weights) # replace the group attention with neighbor attention within the neighbor window.         
        
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : kv_seq_len]
            attn_weights = attn_weights + attention_mask

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

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
