
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890
cuda="0,1,2,3"
num_pro=4

#前500条数据测试正序以及倒序的影响

#baseline
# for answer_index in 1 5 10;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise.py \
#     --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/mdqa_result/mdqa_10_layerwise_documents_$answer_index.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --answer_idx $answer_index \
#     --small_bound 1.59 \
#     --big_bound 1.61 \
#     --steps 6 \

# done

# done    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \


#scale从小到大
for layer_num in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31;
do

for answer_index in 0 5 10;
do
CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise.py \
    --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/mdqa_result/mdqa_10_layerwise_documents_$answer_index.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 300 \
    --batch_size 2 \
    --apply_layers $layer_num \
    --answer_idx $answer_index \
    --small_bound 1.59 \
    --big_bound 1.61 \
    --steps 4 \
    --enable_changed_rope

done
done


for layer_num in {1..30..2}; do
    for answer_index in 0 5 10; do
        CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise.py \
            --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
            --output_path ../result/mdqa_result/mdqa_10_layerwise_documents_$answer_index.json \
            --model_name lmsys/vicuna-7b-v1.5 \
            --seed 42 \
            --sample_num 300 \
            --batch_size 2 \
            --apply_layers "$layer_num,$((layer_num+1))" \
            --answer_idx $answer_index \
            --small_bound 1.59 \
            --big_bound 1.61 \
            --steps 2 \
            --enable_changed_rope
    done
done

for layer_num in {1..29..3}; do
    for answer_index in 0 5 10; do
        CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise.py \
            --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
            --output_path ../result/mdqa_result/mdqa_10_layerwise_documents_$answer_index.json \
            --model_name lmsys/vicuna-7b-v1.5 \
            --seed 42 \
            --sample_num 300 \
            --batch_size 2 \
            --apply_layers "$layer_num,$((layer_num+1)),$((layer_num+2))" \
            --answer_idx $answer_index \
            --small_bound 1.59 \
            --big_bound 1.61 \
            --steps 3 \
            --enable_changed_rope
    done
done

for layer_num in {1..28..4}; do
    for answer_index in 0 5 10; do
        CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise.py \
            --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
            --output_path ../result/mdqa_result/mdqa_10_layerwise_documents_$answer_index.json \
            --model_name lmsys/vicuna-7b-v1.5 \
            --seed 42 \
            --sample_num 300 \
            --batch_size 2 \
            --apply_layers "$layer_num,$((layer_num+1)),$((layer_num+2)),$((layer_num+3))" \
            --answer_idx $answer_index \
            --small_bound 1.59 \
            --big_bound 1.61 \
            --steps 4 \
            --enable_changed_rope
    done
done



for answer_index in 0 5 10;
do
CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise.py \
    --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/mdqa_result/mdqa_10_layerwise_documents_$answer_index.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --seed 42 \
    --sample_num 300 \
    --batch_size 2 \
    --apply_layers "4,5,6,7" \
    --answer_idx $answer_index \
    --small_bound 1.59 \
    --big_bound 1.61 \
    --steps 4 \
    --enable_changed_rope

done

# #scale从大到小
# for answer_index in 1 5;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise.py \
#     --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/mdqa_result/mdqa_10_layerwise_documents_$answer_index.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 300 \
#     --enable_changed_rope \
#     --batch_size 2 \
#     --apply_layers "13,14,15,16,17,18,19,20,21,22" \
#     --answer_idx $answer_index \
#     --small_bound 1.4 \
#     --big_bound 1.8 \
#     --steps 10 \
#     --monotonic_decrease

# done

# #只对前面的层数进行改变
# for start_index in 1.3 1.4 ;
# do
# for enc_index in 1.6 1.7 ;
# do
# for i in 1 3 5 7 10 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 Ms-PoE/inference_qa_layerwise.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 1000 \
#     --enable_mutilevel \
#     --batch_size 4 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12" \
#     --answer_idx $i \
#     --small_bound $start_index \
#     --big_bound $enc_index \
#     --steps 1 \

# # python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response_result.py --input-path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json 

# done  
# done
# done

# #只对中间的层数进行改变
# for start_index in 1.3 1.4;
# do
# for enc_index in 1.6 1.7 ;
# do
# for i in 1 3 5 7 10 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 Ms-PoE/inference_qa_layerwise.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 1000 \
#     --enable_mutilevel \
#     --batch_size 4 \
#     --apply_layers "13,14,15,16,17,18,19,20,21,22" \
#     --answer_idx $i \
#     --small_bound $start_index \
#     --big_bound $enc_index \
#     --steps 1 \

# # python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response_result.py --input-path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json 

# done  
# done
# done

# #只对后边的层数进行改变
# for start_index in 1.3 1.4 ;
# do
# for enc_index in 1.6 1.7 ;
# do
# for i in 1 3 5 7 10 ;  
# do  
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 Ms-PoE/inference_qa_layerwise.py \
#     --input_path Ms-PoE/data/mdqa_10documents.jsonl.gz \
#     --output_path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --seed 42 \
#     --sample_num 1000 \
#     --enable_mutilevel \
#     --batch_size 4 \
#     --apply_layers "21,22,23,24,25,26,27,28,29,30,31" \
#     --answer_idx $i \
#     --small_bound $start_index \
#     --big_bound $enc_index \
#     --steps 1 \

# # python -u Ms-PoE/utils/lost_in_the_middle/eval_qa_response_result.py --input-path Ms-PoE/mdqa_results/mdqa_10_layerwise_documents${i}.json 

# done  
# done
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





