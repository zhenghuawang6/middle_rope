#filter dataset
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lost_in_the_middle.metrics import best_subspan_em
from xopen import xopen
import json
from collections import defaultdict

if __name__ == "__main__":
    file_paths = ["/data/wangzh/middle_rope/src/utils/PositionalHidden/experiments/NQ/model_responses/Llama-2-7b-chat-hf/scale_no_scale/nq-open-20_total_documents_gold_at_0jsonl",
                "/data/wangzh/middle_rope/src/utils/PositionalHidden/experiments/NQ/model_responses/Llama-2-7b-chat-hf/scale_no_scale/nq-open-20_total_documents_gold_at_19jsonl",
                "/data/wangzh/middle_rope/src/utils/PositionalHidden/experiments/NQ/model_responses/Llama-2-7b-chat-hf/scale_no_scale/nq-open-20_total_documents_gold_at_9jsonl"]
    result = []
    result_static = defaultdict(int)
    for i,file_path in enumerate(file_paths):
        with xopen(file_path,"r") as fin:
            for j, line in enumerate(fin):
                input_example = json.loads(line)
                if i == 0:
                    input_example["num_correct"] = best_subspan_em(input_example["model_answer"],input_example["answers"])
                    result.append(input_example)
                else:
                    result[j]["num_correct"] += best_subspan_em(input_example["model_answer"],input_example["answers"])

    #记录开始数据以及结束数据
    start_result = []
    end_result = []
    with xopen(file_paths[0],"r") as fin:
            for j, line in enumerate(fin):
                input_example = json.loads(line)
                start_result.append(input_example)

    with xopen(file_paths[1],"r") as fin:
        for j, line in enumerate(fin):
            input_example = json.loads(line)
            end_result.append(input_example)

    #显示统计结果
    for data in result:
        result_static[data["num_correct"]]+=1
    
    print(f"统计结果为:\n{result_static}")
    
    num_write = 0
    store_result = []
    for i, data in enumerate(result):
            if data["num_correct"] == 3:
                if random.random() < 0.05:
                    store_result.append(i)
                    num_write += 1
            elif data["num_correct"] == 1 or data["num_correct"] == 2:
                if random.random() < 0.35:
                    store_result.append(i)
                    num_write += 1
            else:
                if random.random() < 0.05:
                    store_result.append(i)
                    num_write += 1
    print(f"文件的行数为：{num_write}")


    #进行文件的存储
    with xopen("/data/wangzh/middle_rope/data/mutiqa/generated_data/chat_template_llama_2_7b_chat_first_500.jsonl", "w")as f_start, xopen("/data/wangzh/middle_rope/data/mutiqa/generated_data/chat_template_llama_2_7b_chat_end_500.jsonl", "w") as f_end:
        for i, data in enumerate(result):
            if i in store_result:
                f_start.write(json.dumps(start_result[i]) + "\n")
                f_end.write(json.dumps(end_result[i]) + "\n")

    
