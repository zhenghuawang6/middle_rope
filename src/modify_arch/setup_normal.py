from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from .mutillama import MutiLlamaAttention
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


def setup_models(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir, padding_side='left')
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

class LayerwiseLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        cur_layers = num_layers - config.skip_layers
        assert cur_layers%len(config.layer_scales) == 0, f"无法整除"
        #进行扩充
        scale_list = []
        step = int(cur_layers/len(config.layer_scales))
        for scale in config.layer_scales:
            for i in range(step):
                scale_list.append(scale)
        for i in range(config.skip_layers,num_layers):
            config.small_scale = scale_list[i-config.skip_layers]
            config.big_scale = scale_list[i-config.skip_layers]
            self.model.layers[i].self_attn = MutiLlamaAttention(config)

    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None


def setup_models_layerwise(args):
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir,padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.enable_mutilevel:
        config.layer_scales = args.layer_scales
        config.context_len = args.context_len
        config.skip_layers = args.skip_layers
        model = LayerwiseLlamaForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    return config, tokenizer, model