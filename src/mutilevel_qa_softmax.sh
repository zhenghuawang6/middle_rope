
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890
cuda="7"
num_pro=1

#baseline

for i in 5 ;  
do  
CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../inference_qa_softmax.py \
    --input_path ../data/synthwiki/syn_vicuna-7b-v1.5_3200_random.pickle \
    --output_path ../result/mdqa_result/mdqa_10_softmax_documents${i}.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 500 \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --enable_changed_rope \
    --batch_size 2 \
    --answer_idx $i 
done

# for i in 5 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29510 /data/wzh/paperproject/Ms/Ms-PoE/inference_qa_softmax.py \
#     --input_path ../data/mdqa_10documents.jsonl.gz \
#     --output_path ../mdqa_results/mdqa_10_softmax_documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 500 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29"  \
#     --enable_mutilevel \
#     --batch_size 5 \
#     --answer_idx $i 
# done

# for i in 5 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29510 /data/wzh/paperproject/Ms/Ms-PoE/inference_qa_softmax.py \
#     --input_path ../data/mdqa_10documents.jsonl.gz \
#     --output_path ../mdqa_results/mdqa_10_softmax_documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 500 \
#     --apply_layers "12,13,14,15,16,17,18,19"  \
#     --enable_mutilevel \
#     --batch_size 5 \
#     --answer_idx $i 
# done


#正序
# for i in 1 3 5 7 10 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 Ms-PoE/inference_qa_layerwise.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 1000 \
#     --enable_mutilevel \
#     --batch_size 6 \
#     --answer_idx $i \
#     --small_bound 1.2 \
#     --big_bound 1.8 \
#     --skip_layers 2 \
#     --steps 5 \

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response_result.py --input-path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json 

# done  

# 倒叙
# for i in 1 10 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 Ms-PoE/inference_qa_layerwise.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 1000 \
#     --enable_mutilevel \
#     --batch_size 7 \
#     --answer_idx $i \
#     --small_bound 1.2 \
#     --monotonic_decrease \
#     --big_bound 1.8 \
#     --skip_layers 2 \
#     --steps 3 \

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response_result.py --input-path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json 
# done  


#全是1.5
# for i in 1 3 5 7 10 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --enable_mutilevel \
#     --sample_num 1000 \
#     --batch_size 6 \
#     --answer_idx $i \
#     --small_scale 1.5 \
#     --big_scale 1.5

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents${i}.json 
# done  

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --small_bound 1.2 \
#     --big_bound 1.8 \
#     --steps 32 \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 1 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 3 \
#     --small_scale 0.9 \
#     --big_scale 0.9 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 5 \
#     --small_scale 0.9 \
#     --big_scale 0.9 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 7 \
#     --small_scale 0.9 \
#     --big_scale 0.9 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 10 \
#     --small_scale 0.9 \
#     --big_scale 0.9 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 1 \
#     --small_scale 0.8 \
#     --big_scale 0.8 \
#     --context_len 512

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 3 \
#     --small_scale 0.8 \
#     --big_scale 0.8 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 5 \
#     --small_scale 0.8 \
#     --big_scale 0.8 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json


# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 7 \
#     --small_scale 0.8 \
#     --big_scale 0.8 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29505 Ms-PoE/inference_qa.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx 10 \
#     --small_scale 0.8 \
#     --big_scale 0.8 \
#     --context_len 512
# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.json



# CUDA_VISIBLE_DEVICES="4,5" accelerate launch --num_processes=2 --main_process_port=29505 Ms-PoE/inference.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents1.jsonl\
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 4 \
#     --answer_idx 1

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents1.jsonl

# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" accelerate launch --num_processes=6 Ms-PoE/inference.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents3.jsonl\
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 4 \
#     --answer_idx 3

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents3.jsonl

# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" accelerate launch --num_processes=6 Ms-PoE/inference.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents5.jsonl\
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 4 \
#     --answer_idx 5

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents5.jsonl

# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" accelerate launch --num_processes=6 Ms-PoE/inference.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents7.jsonl\
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 4 \
#     --answer_idx 7

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents7.jsonl

# CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --num_processes=4 Ms-PoE/inference.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10documents10.jsonl\
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --enable_mutilevel \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 40 \
#     --batch_size 4 \
#     --answer_idx 10

# python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response.py --input-path Ms-PoE/mdqa_results/mdqa_10documents10.jsonl

#   --enable_mutilevel \
# CUDA_VISIBLE_DEVICES=0 python -u inference.py \
#     --input_path data/mdqa_10documents.jsonl.gz \
#     --output_path mdqa_results/ours-vicuna_7b-10doc-answer3-ratio1.2to1.8.jsonl \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42\
#     --sample_num 500 \
#     --answer_idx 3 \
#     --enable_ms_poe \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --compress_ratio_min 1.2 \
#     --compress_ratio_max 1.8 
# python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer3-ratio1.2to1.8.jsonl



# CUDA_VISIBLE_DEVICES=0 python -u inference.py \
#     --input_path data/mdqa_10documents.jsonl.gz \
#     --output_path mdqa_results/ours-vicuna_7b-10doc-answer5-ratio1.2to1.8.jsonl \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42\
#     --sample_num 500 \
#     --answer_idx 5 \
#     --enable_ms_poe \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --head_type "normal" \
#     --compress_ratio_min 1.2 \
#     --compress_ratio_max 1.8 
# python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer5-ratio1.2to1.8.jsonl


# CUDA_VISIBLE_DEVICES=0 python -u inference.py \
#     --input_path data/mdqa_10documents.jsonl.gz \
#     --output_path mdqa_results/ours-vicuna_7b-10doc-answer7-ratio1.2to1.8.jsonl \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42\
#     --sample_num 500 \
#     --answer_idx 7 \
#     --enable_ms_poe \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --head_type "normal" \
#     --compress_ratio_min 1.2 \
#     --compress_ratio_max 1.8 
# python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer7-ratio1.2to1.8.jsonl


# CUDA_VISIBLE_DEVICES=0 python -u inference.py \
#     --input_path data/mdqa_10documents.jsonl.gz \
#     --output_path mdqa_results/ours-vicuna_7b-10doc-answer10-ratio1.2to1.8.jsonl \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42\
#     --sample_num 500 \
#     --answer_idx 10 \
#     --enable_ms_poe \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --head_type "normal" \
#     --compress_ratio_min 1.2 \
#     --compress_ratio_max 1.8 
# python -u utils/lost_in_the_middle/eval_qa_response.py --input-path mdqa_results/ours-vicuna_7b-10doc-answer10-ratio1.2to1.8.jsonl





