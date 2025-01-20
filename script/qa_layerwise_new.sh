cuda="0,1,2"
num_pro=3
for answer_idx in 5 7 10;
do
CUDA_VISIBLE_DEVICES=$cuda accelerate launch --num_processes=$num_pro --main_process_port=29509 ../src/inference_qa_layerwise_new.py \
    --input_path ../data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --output_path ../result/layerwise/mdqa_10documents_${answer_idx}.json \
    --model_name lmsys/vicuna-7b-v1.5 \
    --apply_layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --seed 42 \
    --sample_num 500 \
    --batch_size 2 \
    --answer_idx ${answer_idx} \
    --enable_changed_rope

done
