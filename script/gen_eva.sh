CUDA_VISIBLE_DEVICES=0 python ../src/utils/evolution/evaluate_layerwise.py \
    --model_name lmsys/vicuna-7b-v1.5 \
    --input_path ../data/mutiqa/generated_data/nq_10_search_data.jsonl.gz \
    --output_path ../result/layerwise/mdqa_10documents.json \
    --sample_num 500 \
    --batch_size 4 \
    --num_doc 10 \
    --host 127.0.0.1 \
    --port 38947