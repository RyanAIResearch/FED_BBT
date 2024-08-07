import os

local_pop = [5]
kshot = 40
perturb_rate = [0.8]

for lp in local_pop:
    for pr in perturb_rate:
        os.system(
            f'CUDA_VISIBLE_DEVICES=0 python FedBBT_larger_global_pop.py \
        --task_name "dbpedia" \
        --n_prompt_tokens 50 \
        --intrinsic_dim 500 \
        --k_shot {kshot} \
        --device "cuda:0" \
        --seed 42 \
        --loss_type "ce" \
        --cat_or_add "add" \
        --local_iter 8 \
        --frac 1 \
        --local_popsize {lp} \
        --iid 0 \
        --perturb_rate {pr} \
        --perturb 1 \
        --model_name "llama2" \
        --llama_causal 1 \
            > results/llama/dbpedia/larger_global_pop_new_sigma/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr} 2>&1 &'
        )

for lp in local_pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=1 python FedBBT_larger_global_pop.py \
    --task_name "dbpedia" \
    --n_prompt_tokens 50 \
    --intrinsic_dim 500 \
    --k_shot {kshot} \
    --device "cuda:0" \
    --seed 42 \
    --loss_type "ce" \
    --cat_or_add "add" \
    --local_iter 8 \
    --frac 1 \
    --local_popsize {lp} \
    --iid 0 \
    --perturb 0 \
    --model_name "llama2" \
    --llama_causal 1 \
        > results/llama/dbpedia/larger_global_pop_new_sigma/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_noperturb 2>&1 &'
    )


for lp in local_pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=2 python FedBBT_larger_global_pop.py \
    --task_name "dbpedia" \
    --n_prompt_tokens 50 \
    --intrinsic_dim 500 \
    --k_shot {kshot} \
    --device "cuda:0" \
    --seed 42 \
    --loss_type "ce" \
    --cat_or_add "add" \
    --local_iter 8 \
    --frac 1 \
    --local_popsize {lp} \
    --iid 1 \
    --model_name "llama2" \
    --llama_causal 1 \
        > results/llama/dbpedia/larger_global_pop_new_sigma/fl_iid_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
    )


for lp in local_pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=3 python FedBBT_larger_global_pop.py \
    --task_name "dbpedia" \
    --n_prompt_tokens 50 \
    --intrinsic_dim 500 \
    --k_shot {kshot} \
    --device "cuda:0" \
    --seed 42 \
    --loss_type "ce" \
    --cat_or_add "add" \
    --local_iter 8 \
    --frac 1 \
    --local_popsize {lp} \
    --iid 1 \
    --model_name "llama2" \
    --llama_causal 1 \
    --norm_prompt 1 \
    --prompt_norm_threshold 15 \
    --prompt_norm_threshold_upper 22 \
        > results/llama/dbpedia/larger_global_pop_new_sigma/fl_iid_kshot{kshot}_frac1_lpop{lp}_norm15_upper22 2>&1 &'
    )

for lp in local_pop:
    for pr in perturb_rate:
        os.system(
            f'CUDA_VISIBLE_DEVICES=4 python FedBBT_larger_global_pop.py \
        --task_name "dbpedia" \
        --n_prompt_tokens 50 \
        --intrinsic_dim 500 \
        --k_shot {kshot} \
        --device "cuda:0" \
        --seed 42 \
        --loss_type "ce" \
        --cat_or_add "add" \
        --local_iter 8 \
        --frac 1 \
        --local_popsize {lp} \
        --iid 0 \
        --perturb_rate {pr} \
        --perturb 1 \
        --norm_prompt 1 \
        --prompt_norm_threshold 15 \
        --prompt_norm_threshold_upper 22 \
            > results/llama/dbpedia/larger_global_pop_new_sigma/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr}_norm15_upper22 2>&1 &'
        )
    