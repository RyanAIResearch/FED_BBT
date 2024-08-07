import os

kshot = 40

# os.system(
#     f'CUDA_VISIBLE_DEVICES=5 python FedTuning.py \
# --task_name "sst2" \
# --n_prompt_tokens 50 \
# --k_shot {kshot} \
# --device "cuda:0" \
# --seed 42 \
# --loss_type "ce" \
# --cat_or_add "add" \
# --local_epochs 5 \
# --local_iter 8 \
# --frac 1 \
# --iid 0 \
#     > results/sst2/fedtuning/fl_noniid_alpha0.5_kshot{kshot}_lr1e-2_frac1 2>&1 &'
# )


# os.system(
#     f'CUDA_VISIBLE_DEVICES=5 python FedTuning.py \
# --task_name "sst2" \
# --n_prompt_tokens 50 \
# --k_shot {kshot} \
# --device "cuda:0" \
# --seed 42 \
# --loss_type "ce" \
# --cat_or_add "add" \
# --local_epochs 5 \
# --frac 1 \
# --iid 1 \
#     > results/sst2/fedtuning/fl_iid_kshot{kshot}_lr1e-2_frac1 2>&1 &'
# )

os.system(
    f'CUDA_VISIBLE_DEVICES=5 python FedTuning.py \
--task_name "sst2" \
--n_prompt_tokens 50 \
--k_shot {kshot} \
--device "cuda:0" \
--seed 42 \
--loss_type "ce" \
--cat_or_add "add" \
--local_epochs 5 \
--frac 1 \
--iid 1 \
--batch_size 4 \
--model_name "llama2" \
--llama_causal 1 \
    > results/llama/sst2/fedtuning/fl_iid_kshot{kshot}_lr1e-2_frac1 2>&1 &'
)

os.system(
    f'CUDA_VISIBLE_DEVICES=6 python FedTuning.py \
--task_name "sst2" \
--n_prompt_tokens 50 \
--k_shot {kshot} \
--device "cuda:0" \
--seed 42 \
--loss_type "ce" \
--cat_or_add "add" \
--local_epochs 5 \
--frac 1 \
--iid 0 \
--batch_size 4 \
--model_name "llama2" \
--llama_causal 1 \
    > results/llama/sst2/fedtuning/fl_noniid_alpha0.5_kshot{kshot}_lr1e-2_frac1 2>&1 &'
)