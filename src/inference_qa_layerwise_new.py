
'''
benchmarking MDQA  能够指定正确答案在哪一个部分
'''
import os
import pdb
import json
import math
import torch
import logging
import sys
import warnings
import pickle
import argparse
import dataclasses
import numpy as np
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from rouge import Rouge
from xopen import xopen
import logging
from copy import deepcopy
from utils.log_utils.utils import get_git_commit_hash
accelerator = Accelerator()

warnings.filterwarnings('ignore')

from utils.lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
    get_qa_prompt_index,
    get_qa_prompt_only_true_index,
    get_synthwiki_qa_prompt
)
from utils.lost_in_the_middle.eval_qa_response import evaluate_qa
from modify_arch.setup_layerwise_new import setup_models_layerwise

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def Bezier(ts, points):
    n = points.shape[0] - 1  # n可以取到
     # 基函数结果
    result = []
    for t in ts:
        res = 0
        c = 1 
        for i in range(n+1):  # n可以取到
            if i > 0:
                c = c * (n - i + 1) / i  # 更新贝塞尔基函数的结果系数
            _1_t = (1-t)**i  # (1-t)^i
            _t = t**(n-i)  # t^(n-i)
            res += c * _1_t * _t * points[i]
        result.append(res)
    return result


def format_instruct_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    PROMPT_FOR_GENERATION = "{intro}\n{instruction_key}\n{instruction}\n{response_key}\n".format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction=instruction,
        response_key=RESPONSE_KEY,
    )
    return PROMPT_FOR_GENERATION

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--apply_layers", type=str, default="")
    parser.add_argument('--enable_changed_rope', action='store_true')
    parser.add_argument('--only_true', action='store_true', help='Only use the relevent documenets in the prompt')

    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--answer_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    #创建日志文件
    commit_hash_id =  get_git_commit_hash()
    logging_filename = f"../log/layerwise/{commit_hash_id}_result.log"
    directory = os.path.dirname(logging_filename)
    if not os.path.exists(logging_filename):
        os.makedirs(directory, exist_ok=True)
        with open(logging_filename, 'w') as file:
            file.write("")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO, filename=logging_filename, filemode='a')
    if accelerator.is_main_process:
        logger.info("running %s", " ".join(sys.argv))

    ## set up device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    ## Loading Models
    config, tokenizer, model = setup_models_layerwise(args)
    accelerator.wait_for_everyone()

    ## Loading Dataset
    examples = []
    prompts = []
    all_model_documents = []
    if "pickle" not in args.input_path:
        with xopen(args.input_path, 'r') as f:
            for line in f:
                if line.strip() != '':
                    input_example = json.loads(line)
                    question = input_example["question"]
                    documents = []
                    for ctx in deepcopy(input_example["ctxs"]):
                        documents.append(Document.from_dict(ctx))
                    if not documents:
                        raise ValueError(f"Did not find any documents for example: {input_example}")
                    if args.only_true:
                        prompt = get_qa_prompt_only_true_index(
                            question,
                            documents,
                            mention_random_ordering=False,
                            query_aware_contextualization=False,
                            answer_idx=args.answer_idx
                        )
                    else:
                        prompt = get_qa_prompt_index(
                            question,
                            documents,
                            mention_random_ordering=False,
                            query_aware_contextualization=False,
                            answer_idx=args.answer_idx
                        )
                    if "instruct" in args.model_name:
                        prompt = format_instruct_prompt(prompt)
                    prompts.append(prompt)
                    examples.append(deepcopy(input_example))
                    all_model_documents.append(documents)
    else:
        synthwiki_data = pickle.load(open(args.input_path,"rb"))
        for single_data in synthwiki_data:
            question = single_data["question"]
            documents = single_data["ctxs"]
            prompt = get_synthwiki_qa_prompt(
                question,
                documents,
                mention_random_ordering=False,
                query_aware_contextualization=False,
            )
            if "instruct" in args.model_name:
                prompt = format_instruct_prompt(prompt)
            prompts.append(prompt)
            examples.append(deepcopy(single_data))
            all_model_documents.append(documents)
        

    #对样本进行了采样
    if len(prompts) > args.sample_num:
        prompts = prompts[-args.sample_num:]
        examples = examples[-args.sample_num:]
        all_model_documents = all_model_documents[-args.sample_num:]
        
    if args.enable_changed_rope:
        # 贝塞尔曲线
        layer_num = config.num_hidden_layers
        t = torch.linspace(0,1,layer_num)
        layer_ids = torch.range(start=0,end=32).to(torch.int)
        layer_scales = []
        # 规定四个点
        # points = np.array([[4.95,1.3],[11.6,1.55],[20.65,1.5],[21.65,1.15]])
        points = np.array([[7.15,1.45],[11.60,1.45],[20.65,1.5],[21.8,1.15]])
        Bezier_result = Bezier(t,points)
        for point in Bezier_result:
            layer_scales.append(point[1].item())
        
        # print(f"每层的scale为{layer_scales}")
        
        #得到每层的scale后进行赋值
        model.replace_position_embeddings(layer_ids,layer_scales)

    # print("进行推理。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。")
    responses = []
    with accelerator.split_between_processes(prompts) as sub_prompts:
        with torch.no_grad():
            for batched_prompts in tqdm(chunks(sub_prompts, args.batch_size), total=math.ceil(len(sub_prompts) / args.batch_size)):
                if args.batch_size > 1:
                    input_ids = tokenizer(batched_prompts, add_special_tokens=False, return_tensors='pt', truncation=True, max_length=config.max_position_embeddings, padding=True).input_ids.to(model.device)
                else:
                    input_ids = tokenizer(batched_prompts, add_special_tokens=False, return_tensors='pt', truncation=True, max_length=config.max_position_embeddings).input_ids.to(model.device)
                input_ids = input_ids.cuda()
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=100 + len(input_ids[0]),
                    use_cache=True,
                    return_dict_in_generate=False,
                    do_sample=False
                )
                for i, generated_sequence in enumerate(outputs):
                    text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    if input_ids is None:
                        prompt_length = 0
                    else:
                        prompt_length = len(
                            tokenizer.decode(
                                input_ids[i],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                        )
                    new_text = text[prompt_length:]
                    responses.append(new_text)
                    print("**************************")
                    print(new_text)
                    # print("**************************")

        #从所有的设备上搜集数据
        accelerator.wait_for_everyone()  
        #收集所有的数据
        responses=gather_object(responses)

        if accelerator.is_main_process:
            out_dir=os.path.dirname(args.output_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            with xopen(args.output_path, "w") as f:
                for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
                    output_example = deepcopy(example)
                    # Add some extra metadata to the output example
                    output_example["model_prompt"] = prompt
                    output_example["model_answer"] = response
                    # output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
                    output_example["model"] = args.model_name
                    output_example["model_temperature"] = 0
                    output_example["model_top_p"] = "None"
                    output_example["model_prompt_mention_random_ordering"] = False
                    output_example["model_use_random_ordering"] = False
                    f.write(json.dumps(output_example) + "\n")

            evaluate_qa(args.output_path,None,logger)
