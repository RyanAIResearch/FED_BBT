import os

local_pop = [5]
kshot = 40
perturb_rate = [0.8]

# for lp_idx in range(len(local_pop)):
#     lp = local_pop[lp_idx]
#     for pr in perturb_rate:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=5 python FedBBT_larger_global_pop.py \
#         --task_name "sst2" \
#         --n_prompt_tokens 50 \
#         --intrinsic_dim 1000 \
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
#         --perturb_rate {pr} \
#         --perturb 1 \
#         --model_name "llama2" \
#         --llama_causal 1 \
#         --norm_prompt 1 \
#         --prompt_norm_threshold 15 \
#         --prompt_norm_threshold_upper 22 \
#         --save_prompt 1 \
#             > results/llama/sst2/larger_global_pop_new_sigma_pert/dim1000_fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr}_normprompt_15_upper22 2>&1 &'
#         )

# for lp_idx in range(len(local_pop)):
#     lp = local_pop[lp_idx]
#     for pr in perturb_rate:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=3 python FedPrompt_llama_score.py \
#         --task_name "sst2" \
#         --n_prompt_tokens 50 \
#         --k_shot {kshot} \
#         --device "cuda:0" \
#         --seed 42 \
#         --loss_type "ce" \
#         --cat_or_add "add" \
#         --local_epochs 5 \
#         --frac 1 \
#         --iid 0 \
#         --batch_size 8 \
#         --model_name "llama2" \
#         --epochs 100 \
#         --model_name "llama2" \
#         --llama_causal 0 \
#         --init_prompt_path "hardcode" \
#             > results/llama/sst2/larger_global_pop_new_sigma_pert/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr}_normprompt_15_upper22_ft_score 2>&1 &'
#         )

# for lp in local_pop:
#     for pr in perturb_rate:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=1 python FedBBT_larger_global_pop.py \
#         --task_name "sst2" \
#         --n_prompt_tokens 50 \
#         --intrinsic_dim 500 \
#         --k_shot {kshot} \
#         --device "cuda:0" \
#         --seed 42 \
#         --loss_type "ce" \
#         --cat_or_add "add" \
#         --local_iter 8 \
#         --frac 1 \
#         --iid 0 \
#         --perturb 1 \
#         --model_name "llama2" \
#         --llama_causal 0 \
#         --norm_prompt 1 \
#         --prompt_norm_threshold 15 \
#         --prompt_norm_threshold_upper 22 \
#         --init_score_path \"/workspace/Black-Box-Tuning/results/llama/sst2/fedprompt/fl_iid1_score_state_dict.pt\" \
#             > results/llama/sst2/larger_global_pop_new_sigma_pert/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr}_init_score_norm15_upper22 2>&1 &'
#         )

# for lp in local_pop:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=0 python FedBBT_larger_global_pop.py \
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
#         --iid 1 \
#         --perturb 0 \
#         --model_name "llama2" \
#         --llama_causal 0 \
#         --norm_prompt 1 \
#         --prompt_norm_threshold 15 \
#         --prompt_norm_threshold_upper 22 \
#         --init_score_path \"/workspace/Black-Box-Tuning/results/llama/sst2/fedprompt/fl_iid1_score_state_dict.pt\" \
#             > results/llama/sst2/larger_global_pop_new_sigma_pert/fl_iid_kshot{kshot}_frac1_lpop{lp}_noperturb_init_score_norm15_upper22 2>&1 &'
#         )

# for lp in local_pop:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=6 python FedBBT_larger_global_pop.py \
#         --task_name "sst2" \
#         --n_prompt_tokens 50 \
#         --intrinsic_dim 1000 \
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
#         --iid 1 \
#         --perturb 0 \
#         --model_name "llama2" \
#         --llama_causal 1 \
#         --norm_prompt 1 \
#         --prompt_norm_threshold 15 \
#         --prompt_norm_threshold_upper 22 \
#         --save_prompt 1 \
#             > results/llama/sst2/larger_global_pop_new_sigma_pert/dim1000_fl_iid_kshot{kshot}_frac1_lpop{lp}_noperturb_norm15_upper22 2>&1 &'
#         )

# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=3 python FedPrompt_llama_score.py \
#     --task_name "sst2" \
#     --n_prompt_tokens 50 \
#     --k_shot {kshot} \
#     --device "cuda:0" \
#     --seed 42 \
#     --loss_type "ce" \
#     --cat_or_add "add" \
#     --local_epochs 5 \
#     --frac 1 \
#     --iid 1 \
#     --batch_size 8 \
#     --model_name "llama2" \
#     --epochs 100 \
#     --model_name "llama2" \
#     --llama_causal 0 \
#     --init_prompt_path "hardcode" \
#         > results/llama/sst2/larger_global_pop_new_sigma_pert/fl_iid_kshot{kshot}_frac1_lpop{lp}_noperturb_prompt_ft_score 2>&1 &'
#     )

# for lp in local_pop:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=7 python FedBBT_larger_global_pop.py \
#         --task_name "sst2" \
#         --n_prompt_tokens 50 \
#         --intrinsic_dim 500 \
#         --k_shot 200 \
#         --device "cuda:0" \
#         --seed 42 \
#         --loss_type "ce" \
#         --cat_or_add "add" \
#         --local_iter 1000 \
#         --frac 1 \
#         --local_popsize {lp} \
#         --iid 1 \
#         --perturb 0 \
#         --model_name "llama2" \
#         --llama_causal 1 \
#         --eval_central 1 \
#         --norm_prompt 1 \
#             > results/llama/sst2/central/kshot200_lpop{lp}_normprompt 2>&1 &'
#         )

# for lp in local_pop:
#         os.system(
#             f'CUDA_VISIBLE_DEVICES=2 python FedBBT_larger_global_pop.py \
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
        # )

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
            > results/dbpedia/larger_global_pop_new_sigma/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr} 2>&1 &'
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
        > results/dbpedia/larger_global_pop_new_sigma/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_noperturb 2>&1 &'
    )


# for lp in local_pop:
#     os.system(
#         f'CUDA_VISIBLE_DEVICES=2 python FedBBT_larger_global_pop.py \
#     --task_name "dbpedia" \
#     --n_prompt_tokens 50 \
#     --intrinsic_dim 500 \
#     --k_shot {kshot} \
#     --device "cuda:0" \
#     --seed 42 \
#     --loss_type "ce" \
#     --cat_or_add "add" \
#     --local_iter 8 \
#     --frac 1 \
#     --local_popsize {lp} \
#     --iid 1 \
#         > results/dbpedia/larger_global_pop_new_sigma/fl_iid_kshot{kshot}_frac1_lpop{lp} 2>&1 &'
#     )

for lp in local_pop:
    for pr in perturb_rate:
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
        --iid 0 \
        --perturb_rate {pr} \
        --perturb 1 \
        --norm_prompt 1 \
        --prompt_norm_threshold 15 \
        --prompt_norm_threshold_upper 22 \
            > results/dbpedia/larger_global_pop_new_sigma/fl_noniid_alpha0.5_kshot{kshot}_frac1_lpop{lp}_perturb{pr}_norm15_upper22 2>&1 &'
        )


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