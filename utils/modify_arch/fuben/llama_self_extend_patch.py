# transfromers version 4.32.0
import torch
from transformers.models.llama.modeling_llama import *
import llama_self_extend_patch
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _init_rope(self):
    #正常的index
    self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
            )
    #index放大
    self.rotary_emb1 = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=1)
    #index减少
    self.rotary_emb2 = LlamaLinearScalingRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=1.3)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin) 
    k_embed = (k * cos) + (rotate_half(k) * sin) 
    return q_embed, k_embed


def self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    context_length = 1024,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()

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

    big_cos, big_sin = self.rotary_emb1(value_states, seq_len=kv_seq_len)
    small_cos, small_sin = self.rotary_emb2(value_states, seq_len=kv_seq_len)

    #上下文的index
    big_query_states, big_key_states = apply_rotary_pos_emb(query_states, key_states, big_cos, big_sin, position_ids) # normal attention 
    #query以及后边的index
    small_query_states, small_key_states = apply_rotary_pos_emb(query_states, key_states, small_cos, small_sin, position_ids) # normal attention 
    
    # ********************************************************************************************************************* #
    # _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
    # group_query_states, group_key_states = apply_grouped_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, g_size_1=group_size_1, g_size_2=_re_group_size_2) # grouped attention


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

    # if group_attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #         f" {group_attn_weights.size()}"
    #     )
    
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
        # no cache OR prefill
        neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=big_attn_weights.device)
        neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        group_attention_mask =  torch.tril(torch.ones((q_len-context_length, kv_seq_len), device=big_attn_weights.device),diagonal=context_length)
        neighbor_attention_mask[context_length:, :] -= group_attention_mask

    else:
        raise ValueError("q_len should be 1 or seq_len.")

    merged_attn_weights = torch.where(neighbor_attention_mask.bool(), big_attn_weights, small_attn_weights) # replace the group attention with neighbor attention within the neighbor window. 
    merged_attn_weights = nn.functional.softmax(merged_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 

    # ********************************************************************************************************************* #

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