
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

cuda="7"
num_process=1

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
for seq_len in 2000 5000 7000;
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