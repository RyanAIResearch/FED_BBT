import os
import copy
import time
import random

import torch
# import fitlog
import argparse
import numpy as np
import pickle
from fastNLP import DataSet, DataSetIter, SequentialSampler
from utils import average_weights
from tqdm import tqdm
import transformers

from LMBackwardAPI import LMBackwardAPI
from data_process import data_processor, construct_true_few_shot_data, split_data
from models.prompt_encoder import PromptEncoder


def state_dict_sum(sum_weights, local_weights):
    if sum_weights is None:
        return local_weights
    for key in sum_weights.keys():
        sum_weights[key] += local_weights[key]
    return sum_weights

def state_dict_div(sum_weights, m):
    for key in sum_weights.keys():
        sum_weights[key] = torch.div(sum_weights[key], m)
    return sum_weights



parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='roberta-large',
                    choices=['roberta-base', 'roberta-large',
                             'bert-base-uncased', 'bert-large-uncased',
                             'google/electra-base-generator', 'google/electra-large-generator',
                             'facebook/bart-base', 'facebook/bart-large',
                             't5-small', 't5-base', 't5-large', 't5-3b',
                             'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                             'fnlp/cpt-large', 'llama2'], type=str)
parser.add_argument("--task_name", default='sst2', type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--bound", default=0, type=int)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--alpha", default=1, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--alg", default='CMA', type=str)
parser.add_argument("--random_proj", default='normal', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--loss_type", default='ce', type=str)
parser.add_argument("--cat_or_add", default='add', type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
# fl args
parser.add_argument("--num_users", default=10, type=int)
parser.add_argument("--iid", default=1, type=int)
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--local_epochs", default=5, type=int)
parser.add_argument("--frac", default=1, type=float)
parser.add_argument("--local_popsize", default=20, type=int)
parser.add_argument("--global_popsize", default=200, type=int)
parser.add_argument("--local_iter", default=8, type=int)
parser.add_argument("--alpha_dir", default=0.5, type=float)
parser.add_argument("--lstm_dropout", type=float, default=0.0)
parser.add_argument("--p_tuning", default=0, type=int)
parser.add_argument("--llama_causal", default=0, type=int)
parser.add_argument("--init_score_path", default=None, type=str)
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    help='Path to your onnx model.'
)
args = parser.parse_args()

model_name = args.model_name
task_name = args.task_name
n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
bound = args.bound
sigma = args.sigma
alpha = args.alpha

if args.local_popsize > 0:
    args.local_popsize = args.local_popsize
else:
    args.local_popsize = 4 + 3 * np.log(intrinsic_dim)

args.global_popsize = int(args.frac*args.num_users*args.local_popsize)

device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
cat_or_add = args.cat_or_add
parallel = args.parallel
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path

if inference_framework not in ['pt', 'ort']:
    raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
if inference_framework == 'ort':
    assert onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
    assert os.path.exists(onnx_model_path), f'In valid onnx model path `{onnx_model_path}`'

# fixed hyper-params
if cat_or_add == 'add':
    init_prompt_path = None
else:
    init_prompt_path = './nli_base_prompt.pt'



random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



# Initialize API

model_forward_api = LMBackwardAPI(
    args = args,
    init_prompt_path=init_prompt_path
)

global_weights = copy.deepcopy(model_forward_api.model.state_dict())

global_api_setting = model_forward_api.client_record()

client_api_setting_list = {} 
for i in range(args.num_users):
    client_api_setting_list[i] = model_forward_api.client_record()

# Initialize data processor

data_processor = data_processor(args)


data_bundle = data_processor.get_data()
if task_name in ['agnews', 'yelpp', 'dbpedia', 'snli']:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('test')
else:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('validation')

train_data, dev_data = construct_true_few_shot_data(args, train_data, k_shot)

for ds in [train_data, dev_data, test_data]:
    ds.set_pad_val('input_ids', data_processor.tokenizer.pad_token_id if data_processor.tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)
print('# of train data: {}'.format(len(train_data)))
print('Example:')
print(train_data[0])
print('\n# of dev data: {}'.format(len(dev_data)))
print('Example:')
print(dev_data[0])
print('\n# of test data: {}'.format(len(test_data)))
print('Example:')
print(test_data[0])


# Split dataset
user_dict_train, user_dict_dev = split_data(args, train_data, dev_data)



test_acc = model_forward_api.eval(prompt_embedding=None, test_data=test_data)
print('Global test acc: {}'.format(round(test_acc, 4)))
global_api_setting = model_forward_api.client_record()



for e in range(args.epochs):

    print(f"Global epoch {e}...")
    m = max(int(args.frac * args.num_users), 1)
    #sample users for training
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    local_weights_sum = None
    for idx in idxs_users:
        model_forward_api.model.load_state_dict(global_weights)
        optimizer = transformers.AdamW(params=model_forward_api.model.parameters(), lr=5e-5)
        
        model_forward_api.load_client_record(client_api_setting_list[idx])
        # initialize local data
        train_sample_idxs, dev_sample_idxs = user_dict_train[idx], user_dict_dev[idx]
        print(f"Client {idx} execute local training on {len(train_sample_idxs)} samples...")

        local_train_data = train_data[np.array(list(train_sample_idxs)).tolist()]
        local_dev_data = dev_data[np.array(list(dev_sample_idxs)).tolist()]

        dataloader_train = DataSetIter(local_train_data, batch_size=16, sampler=SequentialSampler())


        model_forward_api.set_dataset(local_train_data, local_dev_data)

        # opt = cma.CMAOptions()
        start_time = time.time()

        for le in range(args.local_epochs):
            for batch_x, batch_y in dataloader_train:
                optimizer.zero_grad()
                model_forward_api.zero_grad()
                loss, _ = model_forward_api.eval(None, train_data=batch_x, train_label=batch_y)
                loss.backward()
                optimizer.step()

            print('Local loss @ local epoch {}: {}'.format(le, loss.item()))
            # es.logger.add()  # write data to disc to be plotted
            # es.disp()
        test_acc = model_forward_api.eval(prompt_embedding=None, test_data=test_data)
        print('Local test acc @ epoch {}: {}'.format(e, round(test_acc, 4)))
        end_time = time.time()

        local_weights_sum=state_dict_sum(local_weights_sum, model_forward_api.model.state_dict())

        client_api_setting_list[idx] = model_forward_api.client_record()

    global_weights = state_dict_div(local_weights_sum, m)
    model_forward_api.model.load_state_dict(global_weights)


   
    print('Global evaluate on test data...')
    test_acc = model_forward_api.eval(prompt_embedding=None, test_data=test_data)
    print('Global test acc @ epoch {}: {}'.format(e, round(test_acc, 4)))
    global_api_setting = model_forward_api.client_record()

    # f= open(f'results/{args.task_name}/prompt/fedavg_prompt_iid{args.iid}_alpha{args.alpha_dir}_popsize{args.local_popsize}.pkl', 'wb+')
    # pickle.dump(client_prompt_dict, f)
    # f.close()

