python -u ./make_qa_data_from_retrieval_results.py \
    --input-path nq-open-contriever-msmarco-retrieved-documents.jsonl.gz \
    --num-total-documents 7 \
    --gold-index 0 \
    --output-path generated_data/nq-open-30_total_documents_gold_at_0.jsonl.gz
