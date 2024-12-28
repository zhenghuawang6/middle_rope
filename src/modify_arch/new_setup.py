from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from .new_llamaattention import NewLlamaAttention
import torch
import copy

ChangedLlamaAttention = None

class MutiLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                self.model.layers[layer_idx].self_attn = NewLlamaAttention(config,layer_idx=layer_idx)


def new_setup_models(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir, padding_side='left')
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<custom_pad>'})
        model.resize_token_embeddings(len(tokenizer))

    if args.enable_changed_rope:
        #上下文后边的缩放
        config.narrow_scale = args.narrow_scale
        #上下文前面的缩放
        config.boost_scale = args.boost_scale
        #需要放大间隔的长度
        config.context_len = args.context_len
        #需要处理的层数
        config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
        model = MutiLlamaForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config,torch_dtype=torch.float16)
    else:
        # config.rope_scaling = {"type":"linear","factor":1.5}
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir,config=config,torch_dtype=torch.float16)
    return config, tokenizer, model.eval().to(args.device)
