import os

pop = [10]
kshot = 40
rs = [50, 66, 88]

for p in pop:
    for r in rs:
        os.system(
            f'CUDA_VISIBLE_DEVICES=6 nohup python bbt.py \
        --task_name "sst2" \
        --n_prompt_tokens 50 \
        --intrinsic_dim 500 \
        --k_shot {kshot} \
        --device "cuda:0" \
        --seed {r} \
        --loss_type "ce" \
        --cat_or_add "add" \
        --budget 8000 \
        --print_every 50 \
        --eval_every 10 \
        --popsize {p} \
            > results/sst2/central_kshot{kshot}_lpop{p}_rs{r} 2>&1 &'
        )