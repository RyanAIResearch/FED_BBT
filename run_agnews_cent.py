import os

pop = [10]
kshot = 500

for p in pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=7 nohup python bbt.py \
    --task_name "agnews" \
    --n_prompt_tokens 50 \
    --intrinsic_dim 500 \
    --k_shot {kshot} \
    --device "cuda:0" \
    --seed 42 \
    --loss_type "ce" \
    --cat_or_add "add" \
    --budget 8000 \
    --print_every 50 \
    --eval_every 10 \
    --popsize {p} \
        > results/agnews/central_kshot{kshot}_lpop{p} 2>&1 &'
    )