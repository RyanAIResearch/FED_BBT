import os

local_pop = [5]
kshot = -1
perturb_rate = [0.8]

for lp_idx in range(len(local_pop)):
    lp = local_pop[lp_idx]
    for pr in perturb_rate:
        os.system(
            f'CUDA_VISIBLE_DEVICES=4 python FedBBT_larger_global_pop_multiple_batch.py \
        --task_name "sst2" \
        --n_prompt_tokens 50 \
        --intrinsic_dim 500 \
        --k_shot {kshot} \
        --device "cuda:0" \
        --seed 42 \
        --loss_type "ce" \
        --cat_or_add "add" \
        --local_iter 1 \
        --frac 1 \
        --local_popsize {lp} \
        --iid 0 \
        --perturb_rate {pr} \
        --perturb 1 \
            > results/sst2/larger_global_pop_new_sigma_pert/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr} 2>&1 &'
        )

for lp_idx in range(len(local_pop)):
    lp = local_pop[lp_idx]
    for pr in perturb_rate:
        os.system(
            f'CUDA_VISIBLE_DEVICES=4 python FedBBT_larger_global_pop_multiple_batch.py \
        --task_name "sst2" \
        --n_prompt_tokens 50 \
        --intrinsic_dim 500 \
        --k_shot {kshot} \
        --device "cuda:0" \
        --seed 42 \
        --loss_type "ce" \
        --cat_or_add "add" \
        --local_iter 1 \
        --frac 1 \
        --local_popsize {lp} \
        --iid 1 \
        --perturb_rate {pr} \
        --perturb 1 \
            > results/sst2/larger_global_pop_new_sigma_pert/fl_iid_kshot{kshot}_frac1_lpop{lp}_perturb{pr} 2>&1 &'
        )

# for lp in local_pop:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=2 python FedBBT_larger_global_pop_multiple_batch.py \
#         --task_name "sst2" \
#         --n_prompt_tokens 50 \
#         --intrinsic_dim 500 \
#         --k_shot {kshot} \
#         --device "cuda:0" \
#         --seed 42 \
#         --loss_type "ce" \
#         --cat_or_add "add" \
#         --print_every {lp*8} \
#         --eval_every {lp*8} \
#         --local_iter 8 \
#         --frac 1 \
#         --local_popsize {lp} \
#         --iid 0 \
#         --perturb 0 \
#             > results/sst2/larger_global_pop_new_sigma_pert/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_noperturb 2>&1 &'
#         )


# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=1 python FedBBT_larger_global_pop.py \
#     --task_name "sst2" \
#     --n_prompt_tokens 50 \
#     --intrinsic_dim 500 \
#     --k_shot {kshot} \
#     --device "cuda:0" \
#     --seed 42 \
#     --loss_type "ce" \
#     --cat_or_add "add" \
#     --print_every {lp*8} \
#     --eval_every {lp*8} \
#     --local_iter 8 \
#     --frac 1 \
#     --local_popsize {lp} \
#     --iid 1 \
#     --note "new_sigma" \
#         > results/sst2/larger_global_pop_new_sigma/fl_iid_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )


# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=0 nohup python FedAvg.py \
#     --task_name "sst2" \
#     --n_prompt_tokens 50 \
#     --intrinsic_dim 500 \
#     --k_shot {kshot} \
#     --device "cuda:0" \
#     --seed 42 \
#     --loss_type "ce" \
#     --cat_or_add "add" \
#     --print_every {lp*8} \
#     --eval_every {lp*8} \
#     --local_iter 8 \
#     --frac 1 \
#     --local_popsize {lp} \
#     --iid 0 \
#         > results/sst2/fedavg_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=1 nohup python FedAvg.py \
#     --task_name "sst2" \
#     --n_prompt_tokens 50 \
#     --intrinsic_dim 500 \
#     --k_shot {kshot} \
#     --device "cuda:0" \
#     --seed 42 \
#     --loss_type "ce" \
#     --cat_or_add "add" \
#     --print_every {lp*8} \
#     --eval_every {lp*8} \
#     --local_iter 8 \
#     --frac 1 \
#     --local_popsize {lp} \
#     --iid 1 \
#         > results/sst2/fedavg_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )