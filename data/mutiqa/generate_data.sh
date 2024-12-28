for total_docs in 10 20 30 40 50 60;
do
    for gold in 0 $((total_docs / 2)) $((total_docs - 1));
    do
    python -u ./make_qa_data_from_retrieval_results.py \
        --input-path nq-open-contriever-msmarco-retrieved-documents.jsonl.gz \
        --num-total-documents $total_docs \
        --gold-index $gold \
        --output-path generated_data/nq-open-${total_docs}_total_documents_gold_at_${gold}.jsonl.gz
    done
done