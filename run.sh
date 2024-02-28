# normal evaluation
python main.py mmlu --model_name causal --model_path /gscratch/zlab/llama2/Llama-2-7b-hf --ntrain 1

# prepare inputs for mmlu retrieval
python main.py mmlu_retrieval_prepare --max_eval_seq_len 1024 --ntrain 0 --overwrite_saved_data True

# evaluate with retrieved results
python main.py mmlu_retrieval \
  --model_name causal --model_path /gscratch/zlab/llama2/Llama-2-7b-hf \
  --test_jsonl_with_retrieval /gscratch/scrubbed/rulins/scaling_out/retrieved_results/rpj_wiki_datastore-256_chunk_size-1of1_shards/saved_mmlu_inputs_for_retrieval-0_shots-1024_max_seq_len_retrieved_results.jsonl \
  --concate_k 1 \
  --ntrain 5