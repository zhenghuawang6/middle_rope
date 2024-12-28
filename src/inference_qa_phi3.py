
'''
benchmarking MDQA  能够指定正确答案在哪一个部分
'''
import os
import pdb
import json
import math
import torch
import logging
import warnings
import pickle
import argparse
import dataclasses
import numpy as np
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from rouge import Rouge
import sys
from xopen import xopen
import logging
from copy import deepcopy
from utils.log_utils.utils import get_git_commit_hash
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.lost_in_the_middle.eval_qa_response import evaluate_qa
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing
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

from modify_arch.setup_normal import setup_models

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
    parser.add_argument('--enable_ms_poe', action='store_true')

    parser.add_argument('--only_true', action='store_true', help='Only use the relevent documenets in the prompt')
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--answer_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    commit_hash_id = get_git_commit_hash()
    logging_filename = f"../log/phi3/{commit_hash_id}_result.log"
    directory = os.path.dirname(logging_filename)
    os.makedirs(directory, exist_ok=True)
    if not os.path.exists(logging_filename):
        with open(logging_filename, 'w') as file:
            file.write("")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO, filename=logging_filename, filemode='a')

    if accelerator.is_main_process:
        logger.info(f"model_name:{args.model_name},dataset:{args.input_path},answer_index:{args.answer_idx}")

    ## set up device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    # 加载模型
    # config, tokenizer, model = setup_models(args)
    #加载llama3
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, cache_dir=args.cache_dir, padding_side='left')

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
            prompts.append(prompt)
            examples.append(deepcopy(single_data))
            all_model_documents.append(documents)
        if "instruct" in args.model_name:
            prompt = format_instruct_prompt(prompt)
    #对样本进行了采样
    if len(prompts) > args.sample_num:
        prompts = prompts[:args.sample_num]
        examples = examples[:args.sample_num]
        all_model_documents = all_model_documents[:args.sample_num]


    # Generate Results
    responses = []
    with accelerator.split_between_processes(prompts) as sub_prompts:
        with torch.no_grad():
            for batched_prompts in tqdm(chunks(sub_prompts, args.batch_size), total=math.ceil(len(sub_prompts) / args.batch_size)):
                if args.batch_size > 1:
                    tok_input = tokenizer(batched_prompts, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=config.max_position_embeddings, padding=True)
                else:
                    tok_input = tokenizer(batched_prompts, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=config.max_position_embeddings)
                tok_input = {key: value.cuda() for key, value in tok_input.items()}
                input_ids = tok_input["input_ids"]
                outputs = model.generate(
                    **tok_input,
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
                    print(new_text)
                    print("--------------------------------")
                    responses.append(new_text)

        #从所有的设备上搜集数据
        accelerator.wait_for_everyone()  
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
                    # output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
                    output_example["model_answer"] = response
                    output_example["model"] = args.model_name
                    output_example["model_temperature"] = 0
                    output_example["model_top_p"] = "None"
                    output_example["model_prompt_mention_random_ordering"] = False
                    output_example["model_use_random_ordering"] = False
                    f.write(json.dumps(output_example) + "\n")

            evaluate_qa(args.output_path,None,logger)
