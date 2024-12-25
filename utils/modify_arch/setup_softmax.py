from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from .mutillama_softmax import MutiLlamaAttention
import torch
import copy

ChangedLlamaAttention = None

class MutiLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                self.model.layers[layer_idx].self_attn = MutiLlamaAttention(config)



def setup_models_softmax(args):
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir,padding_side='left')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if args.enable_changed_rope:
        config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
        model = MutiLlamaForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config,torch_dtype=torch.float16)
    else:
        # config.rope_scaling = {"type":"linear","factor":1.5}
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir,config=config,torch_dtype=torch.float16)
    return config, tokenizer, model.eval().cuda()
