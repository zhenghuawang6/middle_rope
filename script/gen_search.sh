# python ../src/utils/evolution/search.py \
#     --model_name lmsys/vicuna-7b-v1.5 \
#     --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/nq-open-10_total_documents_gold_at_0.jsonl.gz \
#     --output_path ../result/layerwise/evo_mdqa_10documents.json \
#     --sample_num 100 \
#     --batch_size 2 \
#     --num_doc 10 \
#     --output_dir ../log/last_500_part_evolution/ \
#     --cuda_indexs "2,3"

# python ../src/utils/evolution/search.py \
#     --model_name ../download/Llama-2-7b-chat-hf \
#     --input_path /data/wangzh/middle_rope/src/utils/PositionalHidden/experiments/NQ/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz \
#     --output_path ../result/layerwise/evo_mdqa_10documents.json \
#     --sample_num 100 \
#     --batch_size 1 \
#     --num_doc 20 \
#     --output_dir ../log/20nq_last_100/ \
#     --cuda_indexs "1,2,3"


python ../src/utils/evolution/search.py \
    --model_name ../download/Llama-2-7b-chat-hf \
    --input_path /data/wangzh/middle_rope/data/mutiqa/generated_data/chat_template_llama_2_7b_chat_first_500.jsonl \
    --output_path ../result/layerwise/chat_llame_mdqa_10documents.json \
    --sample_num 100 \
    --batch_size 1 \
    --num_doc 20 \
    --output_dir ../log/llama_7b_chat/nq20_first_100 \
    --cuda_indexs "1,0,3"



# python ../src/utils/evolution/search.py \
#     --model_name ../download/StableBeluga-7B \
#     --input_path ../ori_result/StableBeluga-7B/evolution_data.jsonl.gz \
#     --output_path ../result/layerwise/mdqa_10documents.json \
#     --sample_num 500 \
#     --batch_size 1 \
#     --num_doc 10 \
#     --output_dir ../log/part_evolution/StableBeluga_7B\ \
#     --cuda_indexs "1,3"


# python ../src/utils/evolution/search.py \
#     --model_name ../download/Llama-2-7b-chat-hf \
#     --input_path ../ori_result/Llama-2-7b-chat-hf/evolution_data.jsonl.gz \
#     --output_path ../result/layerwise/llamamdqa_10documents.json \
#     --sample_num 500 \
#     --batch_size 1 \
#     --num_doc 10 \
#     --output_dir ../log/part_evolution/llama2_7B \
#     --cuda_indexs "2,3" \
#     --recovery /data/wangzh/middle_rope/log/part_evolution/llama2_7B/log-20250123-005759.json \

