vllm serve Qwen/Qwen2.5-0.5B-Instruct \
	--max-model-len=256 \
	--max-num-seqs=1 \
	--quantization=bitsandbytes
