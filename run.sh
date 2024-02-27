# python main.py mmlu --model_name causal --model_path /gscratch/zlab/llama2/Llama-2-7b-hf

# prepare inputs for mmlu retrieval
python main.py mmlu_retrieval_prepare --max_eval_seq_len 1024 --ntrain 0 --overwrite_saved_data True