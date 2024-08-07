import os

kshot = 40

# os.system(
#     f'CUDA_VISIBLE_DEVICES=4 python FedPrompt.py \
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


# os.system(
#     f'CUDA_VISIBLE_DEVICES=4 python FedPrompt.py \
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
#     > results/sst2/fedprompt/fl_iid_kshot{kshot}_lr1e-2_frac1 2>&1 &'
# )

# os.system(
#     f'CUDA_VISIBLE_DEVICES=1 python FedPrompt.py \
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
# --batch_size 8 \
# --model_name "llama2" \
#     > results/llama/sst2/fedprompt/fl_noniid_alpha0.5_kshot{kshot}_lr1e-2_frac1 2>&1 &'
# )


# os.system(
#     f'CUDA_VISIBLE_DEVICES=3 python FedPrompt.py \
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
# --batch_size 8 \
# --model_name "llama2" \
#     > results/llama/sst2/fedprompt/fl_iid_kshot{kshot}_lr1e-2_frac1 2>&1 &'
# )

# os.system(
#     f'CUDA_VISIBLE_DEVICES=1 python PromptTuning.py \
# --task_name "sst2" \
# --n_prompt_tokens 50 \
# --k_shot {kshot} \
# --device "cuda:0" \
# --seed 42 \
# --loss_type "ce" \
# --cat_or_add "add" \
# --local_epochs 500 \
# --frac 1 \
# --iid 1 \
# --batch_size 8 \
# --model_name "llama2" \
#     > results/llama/sst2/fedprompt/central_kshot{kshot}_frac1 2>&1 &'
# )

# os.system(
#     f'CUDA_VISIBLE_DEVICES=3 python FedPrompt.py \
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
# --batch_size 8 \
# --model_name "llama2" \
# --llama_causal 1 \
#     > results/llama/sst2/fedprompt/fl_iid_kshot{kshot}_lr1e-2_frac1_noscore_norm 2>&1 &'
# )

# os.system(
#     f'CUDA_VISIBLE_DEVICES=5 python FedPrompt.py \
# --task_name "sst2" \
# --n_prompt_tokens 50 \
# --k_shot {kshot} \
# --device "cuda:0" \
# --seed 42 \
# --loss_type "ce" \
# --cat_or_add "add" \
# --local_epochs 5 \
# --frac 1 \
# --iid 0 \
# --batch_size 8 \
# --model_name "llama2" \
# --llama_causal 1 \
#     > results/llama/sst2/fedprompt/fl_noniid_alpha0.5_kshot{kshot}_lr1e-2_frac1_noscore 2>&1 &'
# )

os.system(
    f'CUDA_VISIBLE_DEVICES=1 python FedPrompt_llama_score.py \
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
--batch_size 8 \
--model_name "llama2" \
--epochs 20 \
    > results/llama/sst2/fedprompt/fl_iid_kshot{kshot}_lr1e-2_frac1_noprompt_store 2>&1 &'
)