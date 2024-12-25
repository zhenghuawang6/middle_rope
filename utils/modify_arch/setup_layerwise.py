from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from .mutillama import MutiLlamaAttention
import torch
import copy

# ChangedLlamaAttention = None

# class MutiLlamaForCausalLM(LlamaForCausalLM):
#     def __init__(self, config):
#         super().__init__(config)
#         num_layers = len(self.model.layers)
#         pre_config = copy.deepcopy(config)
#         pre_config.big_scale = 0.9
#         pre_config.small_scale = 0.9
#         for layer_idx in range(num_layers):
#             if layer_idx == 0 or layer_idx ==1:
#                 self.model.layers[layer_idx].self_attn = MutiLlamaAttention(pre_config)
#             if layer_idx in config.apply_layers:
#                 self.model.layers[layer_idx].self_attn = MutiLlamaAttention(config)

#     def _reset(self):
#         for layer_idx in self.config.apply_layers:
#             self.model.layers[layer_idx].self_attn.enable_head_metrics = True
#             self.model.layers[layer_idx].self_attn.head_order = None



# def setup_models(args):
#     global ChangedLlamaAttention
#     config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
#     #测试内插
    
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir,padding_side='left')
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.unk_token
#     if args.enable_mutilevel:
#         # config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
#         config.small_scale = args.small_scale
#         config.big_scale = args.big_scale
#         config.context_len = args.context_len
#         # # LlamaAttention.forward = llama_self_extend_patch.self_extend_forward
#         model = MutiLlamaForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, config=config,torch_dtype=torch.float16)
#     else:
#         config.rope_scaling = {"type":"linear","factor":1.5}
#         model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir,config=config,torch_dtype=torch.float16)
#     # if args.enable_ms_poe:
#     #     print('Using Ms-PoE Positional Embedding')
#     #     config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
#     #     config.compress_ratio_min = args.compress_ratio_min
#     #     config.compress_ratio_max = args.compress_ratio_max
#     #     config.head_type = args.head_type
#     #     print('Compress Ratio: from {} to {}'.format(config.compress_ratio_min, config.compress_ratio_max))
#     #     model = MsPoELlamaForCausalLM.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir)
#     # else:
#     #     print('Using the Baseline Model')
#     #     model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # return config, tokenizer, model.eval()

class LayerwiseLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        scale_list = []
        #对scale进行复制，使之与层数对应
        scale_rep = int(len(config.apply_layers)/len(config.layer_scales))
        for scale in config.layer_scales:
            for i in range(scale_rep):
                scale_list.append(scale)
        for i,layer_num in enumerate(config.apply_layers):
            config.narrow_scale = scale_list[i]
            config.boost_scale = scale_list[i]
            self.model.layers[layer_num].self_attn = MutiLlamaAttention(config)
    
    #尝试对前两层进行间隔的放大
    def process_first_layers(self,config):
        preconfig = copy.deepcopy(config)
        preconfig.big_scale = 0.9
        preconfig.small_scale = 0.9
        self.model.layers[0].self_attn = MutiLlamaAttention(preconfig)
        self.model.layers[1].self_attn = MutiLlamaAttention(preconfig)


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