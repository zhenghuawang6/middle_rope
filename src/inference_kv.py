#!/usr/bin/env python3
"""Given a data file with KV records, get LM retrieval results.

The KV records are used in the exact order that they're given.
"""
import argparse
import json
import logging
import math
import pathlib
import random
import sys
from copy import deepcopy
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xopen import xopen
from modify_arch.setup_normal import setup_models
from utils.lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
    get_qa_prompt_index,
    get_qa_prompt_only_true_index,
    get_kv_retrieval_prompt
)

accelerator = Accelerator()
logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name,
    batch_size,
    gold_index,
    query_aware_contextualization,
    max_new_tokens,
    output_path,
    args
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_ordered_kv_records = []
    did_format_warn = False
    count = 0
    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            if count >= args.sample_num:
                break
            count+=1
            input_example = json.loads(line)
            # Get the prediction for the input example
            ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
            key = input_example["key"]
            value = input_example["value"]
            original_kv_index = ordered_kv_records.index([key, value])
            # Remove the kv to retrieve from its original index
            original_kv = ordered_kv_records.pop(original_kv_index)
            # Insert it at the specified gold index
            ordered_kv_records.insert(gold_index, original_kv)

            kv_prompt = get_kv_retrieval_prompt(
                data=ordered_kv_records, key=key, query_aware_contextualization=query_aware_contextualization
            )

            if "instruct" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an instruct model, applying instruct formatting")
                    did_format_warn = True
                kv_prompt = format_instruct_prompt(kv_prompt)
            prompts.append(kv_prompt)
            examples.append(deepcopy(input_example))
            all_model_ordered_kv_records.append(ordered_kv_records)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, tokenizer, model = setup_models(args)
    model.half().eval().cuda()
    accelerator.wait_for_everyone()

    responses = []
    with accelerator.split_between_processes(prompts) as sub_prompts:
        with torch.no_grad():
            for batched_prompts in tqdm(chunks(sub_prompts, batch_size), total=math.ceil(len(sub_prompts) / batch_size)):
                inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                for i, generated_sequence in enumerate(outputs):
                    text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    if inputs.input_ids[i] is None:
                        prompt_length = 0
                    else:
                        prompt_length = len(
                            tokenizer.decode(
                                inputs.input_ids[i],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                        )
                    new_text = text[prompt_length:]
                    responses.append(new_text)
        #从所有的设备上搜集数据
        accelerator.wait_for_everyone()  
        responses=gather_object(responses)
    if accelerator.is_main_process:
        with xopen(output_path, "w") as f:
            for example, ordered_kv_records, prompt, response in zip(
                examples, all_model_ordered_kv_records, prompts, responses
            ):
                output_example = deepcopy(example)
                # Add some extra metadata to the output example
                output_example["model_prompt"] = prompt
                output_example["model_answer"] = response
                output_example["model"] = model_name
                output_example["model_ordered_kv_records"] = ordered_kv_records
                f.write(json.dumps(output_example) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO, filemode="a", filename="/data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/kv_result.log")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument('--enable_ms_poe', action='store_true')
    parser.add_argument('--enable_mutilevel', action='store_true')
    parser.add_argument("--apply_layers", type=str, default="")
    parser.add_argument("--head_type", type=str, default=None)
    parser.add_argument("--compress_ratio_min", type=float, default=1.2)
    parser.add_argument("--compress_ratio_max", type=float, default=1.8)
    parser.add_argument("--small_scale", type=float, default=1.5)
    parser.add_argument("--big_scale", type=float, default=1.5)
    parser.add_argument("--context_len", type=int, default=512)

    parser.add_argument('--only_true', action='store_true', help='Only use the relevent documenets in the prompt')
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--answer_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--query-aware-contextualization", action="store_true", help="Use query-aware contextualization"
    )
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    if accelerator.is_main_process:
        logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model_name,
        args.batch_size,
        args.answer_idx,
        args.query_aware_contextualization,
        args.max_new_tokens,
        args.output_path,
        args
    )
    # logger.info("finished running %s", sys.argv[0])
