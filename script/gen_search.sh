CUDA_VISIBLE_DEVICES=0,1,2,3 python ../src/utils/evolution/search.py \
    --model_name lmsys/vicuna-7b-v1.5 \
    --input_path ../data/mutiqa/generated_data/nq_10_search_data_can_true.jsonl.gz \
    --output_path ../result/layerwise/mdqa_10documents.json \
    --sample_num 500 \
    --batch_size 1 \
    --num_doc 10 \
    --output_dir ../log/evolution/

