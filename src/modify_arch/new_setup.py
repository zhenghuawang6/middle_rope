from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
# from transformers.models.llama.modeling_llama import LlamaAttention,LlamaSdpaAttention
# from .new_llamaattention_no_continuous1 import NewLlamaAttention
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
        print("添加pad")
        tokenizer.add_special_tokens({'pad_token': '<custom_pad>'})
        # tokenizer.pad

    if args.enable_changed_rope:
        config._attn_implementation = "eager"
        #上下文后边的缩放
        config.narrow_scale = args.narrow_scale
        #上下文前面的缩放
        config.boost_scale = args.boost_scale
        #需要放大间隔的长度
        config.context_len = args.context_len
        #需要处理的层数
        config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
        model = MutiLlamaForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config,torch_dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir,config=config,torch_dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer))

    return config, tokenizer, model.eval().cuda()
