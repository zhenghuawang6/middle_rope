from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from .new_layerwise_attention import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding, LlamaRotaryEmbedding
import torch
import copy

class LayerwiseLlamaForCausalLM_NTK(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        for layer_num in config.apply_layers:
            self.model.layers[layer_num].self_attn = LlamaAttention(config,layer_idx=layer_num)
    
    #传入层数以及对应的scale，进行重新赋值！
    def replace_position_embeddings(self,layer_ids,layer_scales):
        for layer_id, layer_scale in zip(layer_ids, layer_scales):
            self.config.rope_scaling  = {"rope_type":"dynamic","factor":layer_scale}
            modify_position_embedding = LlamaRotaryEmbedding(self.config.head_dim, max_position_embeddings=self.config.max_position_embeddings,device=self.model.layers[layer_id].self_attn.q_proj.weight.device,config=self.config)
            # print(f"layer_id{layer_id}----->device{self.model.layers[layer_id].self_attn.q_proj.weight.device}")
            # modify_position_embedding = LlamaLinearScalingRotaryEmbedding(self.config.head_dim, max_position_embeddings=self.config.max_position_embeddings, scaling_factor=layer_scale, device=self.model.layers[layer_id].self_attn.q_proj.weight.device)
            setattr(self.model.layers[layer_id].self_attn, "rotary_emb", modify_position_embedding)


class LayerwiseLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        for layer_num in config.apply_layers:
            self.model.layers[layer_num].self_attn = LlamaAttention(config,layer_idx=layer_num)
    
    #传入层数以及对应的scale，进行重新赋值！
    def replace_position_embeddings(self,layer_ids,layer_scales):
        for layer_id, layer_scale in zip(layer_ids, layer_scales):
            # print(f"layer_id{layer_id}----->device{self.model.layers[layer_id].self_attn.q_proj.weight.device}")
            modify_position_embedding = LlamaLinearScalingRotaryEmbedding(self.config.head_dim, max_position_embeddings=self.config.max_position_embeddings, scaling_factor=layer_scale, device=self.model.layers[layer_id].self_attn.q_proj.weight.device)
            setattr(self.model.layers[layer_id].self_attn, "rotary_emb", modify_position_embedding)

def setup_models_layerwise(args):
    args.cache_dir = None
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<custom_pad>'})
    if args.enable_changed_rope:
        config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
        # model = LayerwiseLlamaForCausalLM_NTK.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config,torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        model = LayerwiseLlamaForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config,torch_dtype=torch.bfloat16)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    return config, tokenizer, model.eval()