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



python FedTuning.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --k_shot 40 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --frac 1 \
  --iid 1 \
  --alpha_dir 1 \
  --local_epochs 500 \