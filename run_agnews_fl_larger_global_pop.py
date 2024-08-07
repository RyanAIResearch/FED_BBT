import os

local_pop = [5, 10]
kshot = 40

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=4 nohup python FedAvg.py \
#     --task_name "agnews" \
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
#     --alpha_dir 1 \
#         > results/agnews/fedavg_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=5 nohup python FedAvg.py \
#     --task_name "agnews" \
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
#     --alpha_dir 1 \
#         > results/agnews/fedavg_noniid_alpha1_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=0 nohup python FedBBT_larger_global_pop.py \
#     --task_name "agnews" \
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
#     --alpha_dir 1 \
#     --note "new_sigma" \
#     --stimulate 1 \
#         > results/agnews/larger_global_pop_new_sigma_stimulate/fl_noniid_alpha1_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=1 nohup python FedBBT_larger_global_pop.py \
#     --task_name "agnews" \
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
#     --stimulate 1 \
#         > results/agnews/larger_global_pop_new_sigma_stimulate/fl_iid_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )