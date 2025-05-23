
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

cuda="4"
num_process=1
#baseline
# for j in 1.3 1.4 1.5 1.6 1.7 ;
# do
# for i in 1 3 5 7 10 ; 
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --enable_mutilevel \
#     --sample_num 1000 \
#     --batch_size 5 \
#     --answer_idx $i \
#     --small_scale $j \
#     --big_scale $j

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents${i}.json 
# done  
# done

#尝试的是ntk编码
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29506 ../inference_qa_normal.py \
#     --input_path ../data/mutiqa/generated_data/mdqa_10documents.jsonl.gz \
#     --output_path ../mdqa_results/mdqa_10documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --enable_changed_rope \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 1 \
#     --narrow_scale 1.5 \
#     --boost_scale 1.5


CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29506 ../src/inference_qa_normal.py \
    --input_path ../data/synthwiki/syn_vicuna-7b-v1.5_3200_random.pickle \
    --output_path /data/wzh/paperproject/Ms/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --model_name ../download/vicuna-7b-v1.5-16k \
    --apply_layers "17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --seed 42 \
    --enable_changed_rope \
    --sample_num 200 \
    --batch_size 1 \
    --answer_idx 1 \
    --narrow_scale 1.2 \
    --boost_scale 0.9 \

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29506 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --enable_mutilevel \
#     --sample_num 500 \
#     --batch_size 8 \
#     --answer_idx 10 \
#     --small_scale 1.7 \
#     --big_scale 1.7

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29506 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --enable_mutilevel \
#     --sample_num 500 \
#     --batch_size 8 \
#     --answer_idx 10 \
#     --small_scale 1.6 \
#     --big_scale 1.6


#baseline 1.5
# for i in 1 10 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10_15documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --enable_mutilevel \
#     --sample_num 500 \
#     --batch_size 8 \
#     --answer_idx $i \
#     --small_scale 1.5 \
#     --big_scale 1.5
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10_15documents${i}.json
# done
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 1 \
#     --small_scale 1.5 \
#     --big_scale 1.5

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 5 \
#     --answer_idx 3 \
#     --small_scale 1.5\
#     --big_scale 1.5
    
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 1 \
#     --small_scale 1 \
#     --big_scale 1
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 10 \
#     --small_scale 1 \
#     --big_scale 1
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 10 \
#     --small_scale 1 \
#     --big_scale 0.95
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 1 \
#     --small_scale 1 \
#     --big_scale 0.9
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 3 \
#     --small_scale 1 \
#     --big_scale 0.9
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 5 \
#     --small_scale 1 \
#     --big_scale 0.9
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 7 \
#     --small_scale 1 \
#     --big_scale 0.9
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 6 \
#     --answer_idx 10 \
#     --small_scale 1 \
#     --big_scale 0.9
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json