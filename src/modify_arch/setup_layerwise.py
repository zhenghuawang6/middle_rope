from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from .new_layerwise_attention import LlamaAttention
import torch
import copy
class LayerwiseLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer_num in config.apply_layers:
            self.model.layers[layer_num].self_attn = LlamaAttention(config)



def setup_models_layerwise(args):
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir,padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.enable_changed_rope:
        config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
        config.layer_scales = args.layer_scales
        config.context_len = args.context_len
        # config.skip_layers = args.skip_layers
        model = LayerwiseLlamaForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config,torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.float16)
    return config, tokenizer, model.eval().cuda()