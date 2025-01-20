# python ../src/utils/zero_scrolls/experiments/hf/run_hf_model.py --model-name=meta-llama/Llama-2-7b-chat-hf
# python ../src/utils/zero_scrolls/experiments/hf/run_hf_model.py --model-name=meta-llama/Llama-2-13b-chat-hf
# python ../src/utils/zero_scrolls/experiments/hf/run_hf_model.py --model-name=stabilityai/StableBeluga-7B
# python ../src/utils/zero_scrolls/experiments/hf/run_hf_model.py --model-name=stabilityai/StableBeluga-13B
export CUDA_VISIBLE_DEVICES="0,1,2"
python ../src/utils/zero_scrolls/experiments/hf/run_hf_model.py --model-name=lmsys/vicuna-7b-v1.5-16k
python ../src/utils/zero_scrolls/experiments/hf/run_hf_model.py --model-name=lmsys/vicuna-7b-v1.5