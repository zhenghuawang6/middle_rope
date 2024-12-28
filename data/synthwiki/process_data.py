import pandas as pd
import numpy as np
import random
import pickle
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import os

def genJunkContext(contexts, limit, tokenizer):
    random.shuffle(contexts)
    token_count = 0
    sublist = []
    for paragraph in contexts:
        paragraph_tokens = tokenizer.encode(paragraph)
        paragraph_token_count = len(paragraph_tokens)
        if token_count + paragraph_token_count > limit:
            break
        else:
            token_count += paragraph_token_count
            sublist.append(paragraph)
    return sublist

def insertIntoJunk(junk, doc, insert_place):
    if insert_place == 'halfway':
        pos_to_insert = int(np.floor(len(junk)/2))  # type: ignore
    elif insert_place == 'random':
        pos_to_insert = np.random.randint(len(junk))
    elif insert_place == 'first':
        pos_to_insert = 0
    elif insert_place == 'last':
        pos_to_insert = len(junk)-1
    else:
        raise RuntimeError(insert_place)
    junk[pos_to_insert] = doc
    return junk, pos_to_insert


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--input_file', 
                        default='../../data/synthwiki/madlibs1.csv',
                        help='Where questions?')
    parser.add_argument('--junk_size', 
                        default=3200,
                        type=int,
                        help='How much junk context (in tokens)?')
    parser.add_argument('--insert_place', 
                        default='random',
                        help='Should I put real doc at (max_pos / 2) or random?')
    parser.add_argument('--model_name', 
                        default='lmsys/vicuna-7b-v1.5')
    args = parser.parse_args()


    input_file = args.input_file
    junk_size = args.junk_size
    insert_place = args.insert_place
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = pd.read_csv(input_file)


    all_contexts = np.unique(raw['context'].values)
    ##采样多少数据
    re_ordered = raw.sample(frac=1)
    real_context = re_ordered['context'].values
    real_question = re_ordered['question'].values
    real_answer = re_ordered['answer'].values
    result = []
    for q_idx, (question, context, answer) in tqdm(enumerate(zip(real_question, real_context, real_answer)),total=len(real_question),colour="green",desc="dealing with data"):
        junk_contexts = [c for c in all_contexts if c != context]
        context_to_use = genJunkContext(
            junk_contexts, 
            limit=junk_size, 
            tokenizer=tokenizer,
        )
        random.shuffle(context_to_use)
        supp_docs, pos_to_insert = insertIntoJunk(context_to_use, context, insert_place)

        #进行存储
        result.append({"question":question, "ctxs":supp_docs, "answers":answer})
    model_name = model_name.split("/")[-1]
    pickle.dump(result,open(f"./generated_data/syn_{model_name}_{junk_size}_{insert_place}.pickle","wb"))