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
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from xopen import xopen
import logging
import socket
from copy import deepcopy
from utils.log_utils.utils import get_git_commit_hash
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


def get_prompt_data(args):
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
        
    return prompts, examples, all_model_documents

def commpute_metric(model, tokenizer, config, examples, prompts, args):
    responses = []
    prompts = prompts
    with torch.no_grad():
        for batched_prompts in tqdm(chunks(prompts, args.batch_size), total=math.ceil(len(prompts) / args.batch_size)):
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

        out_dir=os.path.dirname(args.output_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with xopen(args.output_path, "w") as f:
            for example, prompt, response in zip(examples, prompts, responses):
                output_example = deepcopy(example)
                # Add some extra metadata to the output example
                output_example["model_prompt"] = prompt
                output_example["model_answer"] = response
                output_example["model"] = args.model_name
                output_example["model_temperature"] = 0
                output_example["model_top_p"] = "None"
                output_example["model_prompt_mention_random_ordering"] = False
                output_example["model_use_random_ordering"] = False
                f.write(json.dumps(output_example) + "\n")

        return evaluate_qa(args.output_path,None,logger)





def main(args):
    buf_size = 4096
    sock = socket.socket()
    sock.connect((args.host, args.port))
    logger.info(f'Connected to Server [host={args.host}, port={args.port}]')

    #初始化模型
    config, tokenizer, model = setup_models_layerwise(args)
    #初始化数据,通过answer_index获取数据
    args.answer_idx = 1
    start_prompts, start_examples, _ = get_prompt_data(args)
    args.answer_idx = args.num_doc
    end_prompts, end_examples, _ = get_prompt_data(args)
    #发送准备好的数据
    sock.send(json.dumps({'model_ready': True}).encode())

    while True:
        msg: dict = json.loads(sock.recv(buf_size).decode())
        if msg.get('finalize', False):
            logger.info(f'Finalized.')
            break
        # 传过来对应的points
        points: list = msg['points']
        # 贝塞尔曲线
        layer_num = config.num_hidden_layers
        t = torch.linspace(0,1,layer_num)
        layer_ids = torch.range(start=0,end=32).to(torch.int)
        layer_scales = []
        # 规定四个点
        points = np.array(points)
        Bezier_result = Bezier(t,points)
        # 通过白塞尔曲线的32个点，来设置每层的scale
        for point in Bezier_result:
            layer_scales.append(point[1].item())
        model.replace_position_embeddings(layer_ids,layer_scales)

        start_score = commpute_metric(model, tokenizer, config, start_examples, start_prompts, args)
        end_score = commpute_metric(model, tokenizer, config, end_examples, end_prompts, args)

        sock.send(json.dumps({'result': [start_score,end_score]}).encode())

    sock.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--num_doc", type=int, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    args = parser.parse_args()
    args.enable_changed_rope = True

    logger = logging.getLogger(__name__)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    
