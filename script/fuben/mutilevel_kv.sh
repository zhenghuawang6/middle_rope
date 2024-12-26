export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

cuda="6,7"
num_process=2
for i in 1 3 5 7 10 ;  
do  
CUDA_VISIBLE_DEVICES=${cuda} accelerate launch --num_processes=${num_process} --main_process_port=29507 Ms-PoE/inference_kv.py \
    --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys2.jsonl.gz \
    --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --seed 42 \
    --sample_num 500 \
    --batch_size 2 \
    --answer_idx 1 \
    --small_scale 1 \
    --context_len 512 \
    --big_scale 1
python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json
done
CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
    --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
    --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --enable_mutilevel \
    --seed 42 \
    --sample_num 500\
    --batch_size 2 \
    --answer_idx 15 \
    --small_scale 1 \
    --context_len 1024 \
    --big_scale 1

    # --enable_mutilevel \

python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
    --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
    --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --enable_mutilevel \
    --seed 42 \
    --sample_num 500\
    --batch_size 2 \
    --answer_idx 30 \
    --small_scale 1 \
    --context_len 512 \
    --big_scale 1

    # --enable_mutilevel \

python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
    --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
    --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --enable_mutilevel \
    --seed 42 \
    --sample_num 500\
    --batch_size 2 \
    --answer_idx 40 \
    --small_scale 1 \
    --context_len 1024 \
    --big_scale 1

    # --enable_mutilevel \                                                                             

python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
    --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
    --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --enable_mutilevel \
    --seed 42 \
    --sample_num 500\
    --batch_size 2 \
    --answer_idx 50 \
    --small_scale 1 \
    --context_len 512 \
    --big_scale 1

    # --enable_mutilevel \

python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 1 \
#     --small_scale 1 \
#     --context_len 1024 \
#     --big_scale 0.8

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500\
#     --batch_size 2 \
#     --answer_idx 15 \
#     --small_scale 1 \
#     --context_len 1024 \
#     --big_scale 0.8

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500\
#     --batch_size 2 \
#     --answer_idx 30 \
#     --small_scale 1 \
#     --context_len 1024 \
#     --big_scale 0.8

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500\
#     --batch_size 2 \
#     --answer_idx 40 \
#     --small_scale 1 \
#     --context_len 512 \
#     --big_scale 0.8

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500\
#     --batch_size 2 \
#     --answer_idx 50 \
#     --small_scale 1 \
#     --context_len 1024 \
#     --big_scale 0.8

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json


# CUDA_VISIBLE_DEVICES="1,2,3,4,5" accelerate launch --num_processes=5 --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 30 \
#     --small_scale 1 \
#     --big_scale 1

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES="1,2,3,4,5" accelerate launch --num_processes=5 --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 30 \
#     --small_scale 1 \
#     --big_scale 0.85

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES="1,2,3,4,5" accelerate launch --num_processes=5 --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 30 \
#     --small_scale 1 \
#     --big_scale 0.85
#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES="1,2,3,4,5" accelerate launch --num_processes=5 --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 30 \
#     --small_scale 1 \
#     --big_scale 0.8

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES="1,2,3,4,5" accelerate launch --num_processes=5 --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 30 \
#     --small_scale 1 \
#     --big_scale 0.4

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES="1,2,3,4,5" accelerate launch --num_processes=5 --main_process_port=29507 Ms-PoE/inference_kv.py \
#     --input_path /data/wzh/paperproject/Ms/Ms-PoE/kv_data/kv-retrieval-50_keys.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 30 \
#     --small_scale 1 \
#     --big_scale 0.35

#     # --enable_mutilevel \

# python -u /data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/evaluate_kv_responses.py --input-path Ms-PoE/mdqa_results/kv_mdqa_10documents1.json