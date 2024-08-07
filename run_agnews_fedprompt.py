import os

kshot = 40

os.system(
    f'CUDA_VISIBLE_DEVICES=5 python FedPrompt.py \
--task_name "agnews" \
--n_prompt_tokens 50 \
--k_shot {kshot} \
--device "cuda:0" \
--seed 42 \
--loss_type "ce" \
--cat_or_add "add" \
--local_epochs 5 \
--local_iter 8 \
--frac 1 \
--iid 0 \
--alpha_dir 1 \
    > results/agnews/fedprompt/fl_noniid_alpha1_kshot{kshot}_lr1e-2_frac1 2>&1 &'
)


os.system(
    f'CUDA_VISIBLE_DEVICES=5 python FedPrompt.py \
--task_name "agnews" \
--n_prompt_tokens 50 \
--k_shot {kshot} \
--device "cuda:0" \
--seed 42 \
--loss_type "ce" \
--cat_or_add "add" \
--local_epochs 5 \
--frac 1 \
--iid 1 \
    > results/agnews/fedprompt/fl_iid_kshot{kshot}_lr1e-2_frac1 2>&1 &'
)
