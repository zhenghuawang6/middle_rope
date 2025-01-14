
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

cuda="3"
num_process=1

# for seq_len in 15000 10000;
# do
# for loc in "first" "last" "halfway" ;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29506 ../src/inference_qa_llama3.py \
#     --input_path ../data/synthwiki/generated_data/syn_Phi-3-mini-128k-instruct_${seq_len}_${loc}.pickle \
#     --output_path ../result/llama3_result/syn_Phi-3-mini-128k-instruct_${seq_len}_${loc}.json \
#     --model_name ../download/Llama-3.1-8B-Instruct \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 200 \
#     --batch_size 1 \

# done
# done

#40个文档的测评
for gold in 1 4 8 12 16 20 24 28 32 36 40 ;
do

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29509 ../src/inference_qa_llama3.py \
    --input_path ../data/mutiqa/generated_data/nq-open-40_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/llama3_result/mdqa_40documents_$gold.json \
    --model_name ../download/Llama-3.1-8B-Instruct \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --seed 42 \
    --sample_num 500 \
    --batch_size 1 \
    --answer_idx $gold \

done

#50个文档的测评
for gold in 1 5 10 15 20 25 30 35 40 45 50 ;
do

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29509 ../src/inference_qa_llama3.py \
    --input_path ../data/mutiqa/generated_data/nq-open-50_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/llama3_result/mdqa_50documents_$gold.json \
    --model_name ../download/Llama-3.1-8B-Instruct \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --seed 42 \
    --sample_num 500 \
    --batch_size 1 \
    --answer_idx $gold \

done




#qa测试
# for total_docs in 10 20 30;
# do
# gold_values=$(echo "1 $((total_docs / 3)) $((total_docs / 2)) $(echo "scale=0; $total_docs / 1.5" | bc) $total_docs")
# for gold in $gold_values;
# do

# CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29509 ../src/inference_qa_llama3.py \
#     --input_path ../data/mutiqa/generated_data/nq-open-${total_docs}_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/llama3_result/mdqa_10documents${i}.json \
#     --model_name ../download/Llama-3.1-8B-Instruct \
#     --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
#     --seed 42 \
#     --sample_num 200 \
#     --batch_size 2 \
#     --answer_idx $gold \

# done
# done

#syn测试
for seq_len in 10000 15000;
do
for loc in "first" "last" "halfway" ;
do

CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_process --main_process_port=29506 ../src/inference_qa_llama3.py \
    --input_path ../data/synthwiki/generated_data/syn_Phi-3-mini-128k-instruct_${seq_len}_${loc}.pickle \
    --output_path ../result/llama3_result/syn_Phi-3-mini-128k-instruct_${seq_len}_${loc}.json \
    --model_name ../download/Llama-3.1-8B-Instruct \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"  \
    --seed 42 \
    --sample_num 200 \
    --batch_size 1 \

done
done