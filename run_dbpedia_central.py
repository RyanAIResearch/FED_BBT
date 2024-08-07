import os

local_pop = [5]
kshot = 40
perturb_rate = [0.8]

for lp in local_pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=5 python FedBBT_larger_global_pop.py \
    --task_name "dbpedia" \
    --n_prompt_tokens 50 \
    --intrinsic_dim 500 \
    --k_shot 200 \
    --device "cuda:0" \
    --seed 42 \
    --loss_type "ce" \
    --cat_or_add "add" \
    --local_iter 1000 \
    --frac 1 \
    --local_popsize {lp} \
    --iid 1 \
    --perturb 0 \
    --model_name "llama2" \
    --llama_causal 1 \
    --eval_central 1 \
        > results/llama/dbpedia/central/kshot200_lpop{lp} 2>&1 &'
    )

for lp in local_pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=6 python FedBBT_larger_global_pop.py \
    --task_name "dbpedia" \
    --n_prompt_tokens 50 \
    --intrinsic_dim 500 \
    --k_shot 200 \
    --device "cuda:0" \
    --seed 42 \
    --loss_type "ce" \
    --cat_or_add "add" \
    --local_iter 1000 \
    --frac 1 \
    --local_popsize {lp} \
    --iid 1 \
    --perturb 0 \
    --eval_central 1 \
        > results/dbpedia/central/kshot200_lpop{lp} 2>&1 &'
    )