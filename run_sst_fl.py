import os

local_pop = [5]
kshot = 40

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=0 python FedBBT.py \
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
#         > results/sst2/schedule_sigma/new_fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=1 python FedBBT.py \
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
#         > results/sst2/schedule_sigma/new_fl_iid_kshot{kshot}_frac1_lpop{lp}_thres6 2>&1 &'
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


for lp in local_pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=0 nohup python FedAvg.py \
    --task_name "sst2" \
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
    --model_name "llama2" \
    --llama_causal 1 \
        > results/llama/sst2/fedavg/noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
    )

for lp in local_pop:
    os.system(
        f'CUDA_VISIBLE_DEVICES=7 nohup python FedAvg.py \
    --task_name "sst2" \
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
        > results/llama/sst2/fedavg/iid_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
    )