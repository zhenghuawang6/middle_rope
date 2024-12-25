#num-keys 上下文中有多少个kv  num-examples 生成多少个例子

python -u ./make_kv_retrieval_data.py \
    --num-keys 10 \
    --num-examples 500 \
    --output-path generated_data/kv-retrieval-10_keys.jsonl
