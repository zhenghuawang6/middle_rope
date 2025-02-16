cuda="0,1,2,3"
num_pro=4
answer_idx=25

#vicuna

# for answer_idx in 5;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new_softmax.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --enable_changed_rope \
#     --is_chat

# done

# for answer_idx in 5;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new_softmax.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11" \
#     --seed 42 \
#     --sample_num 100 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \

# done


# for answer_idx in 5;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new_softmax.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11" \
#     --seed 42 \
#     --sample_num 100 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --enable_changed_rope \
#     # --is_chat

# done

for answer_idx in 5;
do

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new_softmax.py \
    --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "10,11,12,13" \
    --seed 42 \
    --sample_num 100 \
    --batch_size 2 \
    --answer_idx ${answer_idx} \
    --enable_changed_rope \
    # --is_chat

done

for answer_idx in 5;
do

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new_softmax.py \
    --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "22,23,24,25,26,27,28,29,30" \
    --seed 42 \
    --sample_num 100 \
    --batch_size 2 \
    --answer_idx ${answer_idx} \
    --enable_changed_rope \
    # --is_chat

done

# for answer_idx in 5 10 1;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     # --enable_changed_rope

# done


#stable-begula的baseline
# for answer_idx in 1 3 5 7 10;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/baseline_stable_mdqa_10documents_${answer_idx}.json \
#     --model_name /data/wangzh/middle_rope/download/StableBeluga-7B \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \

# done

# llama的baseline
for answer_idx in 10 5 0;
do

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29506 ../src/inference_qa_layerwise_new.py \
    --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/layerwise/baseline_llama_mdqa_10documents_${answer_idx}.json \
    --model_name /data/wangzh/middle_rope/download/StableBeluga-7B \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --seed 42 \
    --sample_num 500 \
    --batch_size 2 \
    --answer_idx ${answer_idx} \
    --is_chat \
    --enable_changed_rope

done

# for location in last halfway first;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/synthwiki/generated_data/syn_vicuna_filter_true_${location}.pickle \
#     --output_path ../result/layerwise/syn_llama_filter_3800_${location}.json \
#     --model_name ../download/Llama-2-7b-chat-hf \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 300 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset syn \
#     # --enable_changed_rope
# done

# for answer_idx in 50 25 0;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/kv/generated_data/kv-retrieval-50_keys_4000.jsonl \
#     --output_path ../result/layerwise/kv_llama_${answer_idx}.json \
#     --model_name ../download/Llama-2-7b-chat-hf \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 200 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset kv \
#     # --enable_changed_rope

# done

#测试chat
# for location in last halfway first;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/synthwiki/generated_data/syn_vicuna_filter_true_${location}.pickle \
#     --output_path ../result/layerwise/syn_llama_filter_3800_${location}.json \
#     --model_name ../download/Llama-2-7b-chat-hf \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 300 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset syn \
#     --is_chat
#     # --enable_changed_rope
# done

# for answer_idx in 50 25 0;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/kv/generated_data/kv-retrieval-50_keys_4000.jsonl \
#     --output_path ../result/layerwise/kv_llama_${answer_idx}.json \
#     --model_name ../download/Llama-2-7b-chat-hf \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 200 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset kv \
#     --is_chat
#     # --enable_changed_rope

# done


# #测试stable
# for answer_idx in 0 5 9;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29506 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/baseline_llama_mdqa_10documents_${answer_idx}.json \
#     --model_name ../download/StableBeluga-7B \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --is_chat
#     # --enable_changed_rope

# done

# for location in last halfway first;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/synthwiki/generated_data/syn_vicuna_filter_true_${location}.pickle \
#     --output_path ../result/layerwise/syn_llama_filter_3800_${location}.json \
#     --model_name ../download/StableBeluga-7B \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 300 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset syn \
#     # --enable_changed_rope
# done

# for answer_idx in 50 25 0;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/kv/generated_data/kv-retrieval-50_keys_4000.jsonl \
#     --output_path ../result/layerwise/kv_llama_${answer_idx}.json \
#     --model_name ../download/StableBeluga-7B \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 200 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset kv \
#     # --enable_changed_rope

# done

#测试chat
# for location in last halfway first;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/synthwiki/generated_data/syn_vicuna_filter_true_${location}.pickle \
#     --output_path ../result/layerwise/syn_llama_filter_3800_${location}.json \
#     --model_name ../download/StableBeluga-7B \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 300 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset syn \
#     --is_chat \
#     --enable_changed_rope
# done

for answer_idx in 49 25 0;
do
CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
    --input_path ../data/kv/generated_data/kv-retrieval-50_keys_4000.jsonl \
    --output_path ../result/layerwise/kv_llama_${answer_idx}.json \
    --model_name ../download/StableBeluga-7B \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --seed 42 \
    --sample_num 200 \
    --batch_size 1 \
    --answer_idx ${answer_idx} \
    --dataset kv \
    --is_chat \
    --enable_changed_rope

done


#llama的baseline 20个文档

# for answer_idx in 20;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29506 ../src/inference_qa_layerwise_new.py \
#     --input_path /data/wangzh/middle_rope/src/utils/PositionalHidden/experiments/NQ/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/baseline_llama_mdqa_20documents_${answer_idx}.json \
#     --model_name /data/wangzh/middle_rope/download/Llama-2-7b-chat-hf \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     # --enable_changed_rope
# done

#syn vicuna 3800 

# for location in first last halfway;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/synthwiki/generated_data/syn_vicuna-7b-v1.5_3800_epoch_5_${location}.pickle \
#     --output_path ../result/layerwise/syn_vicuna_3800_${location}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 1000 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --dataset syn 

# done

#syn vicuna filter后的结果
# for location in last;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/synthwiki/generated_data/syn_vicuna_filter_true_all_${location}.pickle \
#     --output_path ../result/layerwise/syn_vicuna_filter_3800_${location}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 300 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --dataset syn \
#     # --enable_changed_rope
# done

# for location in last halfway first;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/synthwiki/generated_data/syn_vicuna_filter_true_${location}.pickle \
#     --output_path ../result/layerwise/syn_vicuna_filter_3800_${location}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 300 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset syn \
#     --is_chat \
#     --enable_changed_rope 
# done

#kv vicuna 3800 

# for answer_idx in 50 25 0;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/kv/generated_data/kv-retrieval-50_keys_4000.jsonl \
#     --output_path ../result/layerwise/kv_vicuna_${answer_idx}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 200 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     --dataset kv \
#     # --enable_changed_rope

# done


# for answer_idx in 10 7 5 3 1;
# do
# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
#     --model_name /data/wangzh/middle_rope/download/vicuna-13b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 1 \
#     --answer_idx ${answer_idx} \
#     # --enable_changed_rope

# done

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path /data/wangzh/middle_rope/data/kv/generated_data/kv-retrieval-50_keys_2000.jsonl \
#     --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx 30 \
#     --dataset "kv"



# for answer_idx in 49;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path /data/wangzh/middle_rope/data/kv/generated_data/kv-retrieval-50_keys_4000.jsonl \
#     --output_path ../result/layerwise/kv_50_${answer_idx}.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --dataset "kv"

# done


# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path /data/wangzh/middle_rope/data/synthwiki/generated_data/syn_vicuna-7b-v1.5_3000_epoch_5_first.pickle \
#     --output_path ../result/layerwise/syn_3000_epoch5_first.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --dataset "syn"

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path /data/wangzh/middle_rope/data/synthwiki/generated_data/syn_vicuna-7b-v1.5_3000_epoch_5_halfway.pickle \
#     --output_path ../result/layerwise/syn2_3000_epoch5_halfway.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --dataset "syn"

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
#     --input_path /data/wangzh/middle_rope/data/synthwiki/generated_data/syn_vicuna-7b-v1.5_3000_epoch_5_last.pickle\
#     --output_path ../result/layerwise/syn2_3000_epoch5_last.json \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
#     --seed 42 \
#     --sample_num 500 \
#     --batch_size 2 \
#     --answer_idx ${answer_idx} \
#     --dataset "syn"