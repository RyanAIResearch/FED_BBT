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



# python FedBBT.py \
#   --task_name "agnews" \
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
#   --frac 1 \
#   --iid 0 \
#   --alpha_dir 1 \


CUDA_VISIBLE_DEVICES=4 python FedBBT_larger_global_pop_multiple_batch.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot -1 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --local_iter 1 \
  --frac 1 \
  --local_popsize 5 \
  --iid 1 \
  --perturb_rate 0.8 \
  --perturb 1 \