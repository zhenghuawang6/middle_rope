#num-keys 上下文中有多少个kv  num-examples 生成多少个例子

python -u ./make_kv_retrieval_data.py \
    --num-keys 50 \
    --num-examples 1000 \
    --output-path generated_data/kv-retrieval-50_keys_4000.jsonl

