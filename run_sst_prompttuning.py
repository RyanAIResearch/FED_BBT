import os

kshot = 16

# os.system(
#     f'CUDA_VISIBLE_DEVICES=2 python FedPrompt.py \
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
#     > results/sst2/fedprompt/fl_noniid_alpha0.5_kshot{kshot}_lr1e-2_frac1 2>&1 &'
# )


os.system(
    f'CUDA_VISIBLE_DEVICES=1 python PromptTuning.py \
--task_name "sst2" \
--n_prompt_tokens 50 \
--k_shot {kshot} \
--device "cuda:0" \
--seed 42 \
--loss_type "ce" \
--cat_or_add "add" \
--local_epochs 1000 \
--frac 1 \
--iid 1 \
    > results/sst2/fedprompt/local_iid_kshot{kshot}_lr1e-2 2>&1 &'
)
