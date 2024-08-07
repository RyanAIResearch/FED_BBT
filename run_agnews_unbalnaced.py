import os

pop = [5, 10]
kshot = 40

for p in pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=4 nohup python bbt_unbalanced.py \
    --task_name "agnews" \
    --n_prompt_tokens 50 \
    --intrinsic_dim 500 \
    --k_shot {kshot} \
    --device "cuda:0" \
    --seed 42 \
    --loss_type "ce" \
    --cat_or_add "add" \
    --local_iter 8000 \
    --print_every 50 \
    --eval_every 10 \
    --frac 1 \
    --local_popsize {p} \
    --iid 0 \
    --alpha_dir 1 \
        > results/agnews/unbalanced/noniid_kshot{kshot}_lpop{p} 2>&1 &'
    )