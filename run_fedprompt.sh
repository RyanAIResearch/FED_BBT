# python FedBBT.py \
#   --task_name "sst2" \
#   --n_prompt_tokens 50 \
#   --intrinsic_dim 500 \
#   --k_shot 40 \
#   --device "cuda:0" \
#   --seed 42 \
#   --loss_type "ce" \
#   --cat_or_add "add" \
#   --print_every 160 \
#   --eval_every 160 \
#   --local_iter 8 \
#   --frac 1



# CUDA_VISIBLE_DEVICES=1 python FedPrompt.py \
#   --task_name "sst2" \
#   --n_prompt_tokens 50 \
#   --intrinsic_dim 500 \
#   --k_shot 40 \
#   --device "cuda:0" \
#   --seed 42 \
#   --loss_type "ce" \
#   --cat_or_add "add" \
#   --frac 1 \
#   --iid 1 \
#   --local_epochs 200 \
#   --alpha_dir 1 \
#   --p_tuning 1 \
#   --model_name 'llama2' \
#   --batch_size 8 \

CUDA_VISIBLE_DEVICES=6 python ManualPrompt.py \
  --task_name "sst2" \
  --n_prompt_tokens 0 \
  --intrinsic_dim 500 \
  --k_shot -1 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --frac 1 \
  --iid 1 \
  --local_epochs 200 \
  --alpha_dir 1 \
  --p_tuning 1 \
  --model_name 'llama2' \
  --batch_size 1 \
  --llama_causal 1 \