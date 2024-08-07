import os
import copy
import time
import random

import torch
# import fitlog
import argparse
import numpy as np
import cma
import pickle

from LMForwardAPI import LMForwardAPI, ClientLMForwardAPI
from data_process import data_processor, construct_true_few_shot_data, split_data

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='roberta-large',
                    choices=['roberta-base', 'roberta-large',
                             'bert-base-uncased', 'bert-large-uncased',
                             'google/electra-base-generator', 'google/electra-large-generator',
                             'facebook/bart-base', 'facebook/bart-large',
                             't5-small', 't5-base', 't5-large', 't5-3b',
                             'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                             'fnlp/cpt-large'], type=str)
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
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--frac", default=1, type=float)
parser.add_argument("--local_popsize", default=20, type=int)
parser.add_argument("--global_popsize", default=200, type=int)
parser.add_argument("--local_iter", default=8, type=int)
parser.add_argument("--alpha_dir", default=0.5, type=float)
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


args.bbt_version = 'bbt'


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



# Initialize API

model_forward_api = LMForwardAPI(
    args = args,
    init_prompt_path=init_prompt_path
)

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


# initialize global es
cma_opts = {
        'seed': seed,
        'popsize': args.global_popsize,
        'maxiter': args.epochs,
        'verbose': -1,
    }
if bound > 0:
    cma_opts['bounds'] = [-1 * bound, 1 * bound]
global_es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigma, inopts=cma_opts)

print('Global es evaluate on test data...')
global_api_setting['best_prompt'] = global_es.mean
model_forward_api.load_client_record(global_api_setting)
test_acc = model_forward_api.eval(test_data=test_data)
print('Global test acc: {}'.format(round(test_acc, 4)))
global_api_setting = model_forward_api.client_record()

client_prompt_dict = {}
for c in range(args.num_users):
    client_prompt_dict[c] = [copy.deepcopy(global_es.mean)]
server_prompts = [copy.deepcopy(global_es.mean)]

client_fitnesses_dict = {}
for c in range(args.num_users):
    client_fitnesses_dict[c] = []

schedule_start = False

for e in range(args.epochs):

    global_solutions = []
    global_fitnesses = []

    client_sigmas = {}

    print(f"Global epoch {e}...")
    m = max(int(args.frac * args.num_users), 1)
    #sample users for training
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    for idx in idxs_users:
        
        model_forward_api.load_client_record(client_api_setting_list[idx])
        # initialize local data
        train_sample_idxs, dev_sample_idxs = user_dict_train[idx], user_dict_dev[idx]
        print(f"Client {idx} execute local training on {len(train_sample_idxs)} samples...")
        if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            local_train_data = {
                'input_ids': torch.tensor(train_data['input_ids'].get(train_sample_idxs)),
                'attention_mask': torch.tensor(train_data['attention_mask'].get(train_sample_idxs)),
                'decoder_input_ids': torch.tensor(train_data['decoder_input_ids'].get(train_sample_idxs)),
                'decoder_attention_mask': torch.tensor(train_data['decoder_attention_mask'].get(train_sample_idxs)),
                'labels': torch.tensor(train_data['labels'].get(train_sample_idxs)),
            }
            local_dev_data = {
                'input_ids': torch.tensor(dev_data['input_ids'].get(dev_sample_idxs)),
                'attention_mask': torch.tensor(dev_data['attention_mask'].get(dev_sample_idxs)),
                'decoder_input_ids': torch.tensor(dev_data['decoder_input_ids'].get(dev_sample_idxs)),
                'decoder_attention_mask': torch.tensor(dev_data['decoder_attention_mask'].get(dev_sample_idxs)),
                'labels': torch.tensor(dev_data['labels'].get(dev_sample_idxs)),
            }
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            local_train_data = {
                'input_ids': torch.tensor(train_data['input_ids'].get(train_sample_idxs)),
                'attention_mask': torch.tensor(train_data['attention_mask'].get(train_sample_idxs)),
                'labels': torch.tensor(train_data['labels'].get(train_sample_idxs)),
            }
            local_dev_data = {
                'input_ids': torch.tensor(dev_data['input_ids'].get(dev_sample_idxs)),
                'attention_mask': torch.tensor(dev_data['attention_mask'].get(dev_sample_idxs)),
                'labels': torch.tensor(dev_data['labels'].get(dev_sample_idxs)),
            }
        else:
            local_train_data = {
                'input_ids': torch.tensor(train_data['input_ids'].get(train_sample_idxs)),
                'attention_mask': torch.tensor(train_data['attention_mask'].get(train_sample_idxs)),
                'mask_pos': torch.tensor(train_data['mask_pos'].get(train_sample_idxs)),
                'labels': torch.tensor(train_data['labels'].get(train_sample_idxs)),
            }
            local_dev_data = {
                'input_ids': torch.tensor(dev_data['input_ids'].get(dev_sample_idxs)),
                'attention_mask': torch.tensor(dev_data['attention_mask'].get(dev_sample_idxs)),
                'mask_pos': torch.tensor(dev_data['mask_pos'].get(dev_sample_idxs)),
                'labels': torch.tensor(dev_data['labels'].get(dev_sample_idxs)),
            }



        local_es = global_es._copy_light(inopts={'seed': seed, 'maxiter':args.local_iter, 'popsize':args.local_popsize})
        print('Population Size: {}'.format(local_es.popsize))
        print('{} Evaluation.'.format('Parallel' if parallel else 'Serial'))
        if parallel:
            # expand training data to a larger batch for parallel evaluation
            train_data['input_ids'] = train_data['input_ids'].repeat(local_es.popsize, 1)
            train_data['attention_mask'] = train_data['attention_mask'].repeat(local_es.popsize, 1)
            train_data['mask_pos'] = train_data['mask_pos'].repeat(local_es.popsize)
            train_data['labels'] = train_data['labels'].repeat(local_es.popsize)

        model_forward_api.set_dataset(local_train_data, local_dev_data)

        # opt = cma.CMAOptions()
        local_sigmas = []
        start_time = time.time()
        while not local_es.stop():
            local_sigmas.append(local_es.sigma)
            solutions = local_es.ask()
            if parallel:
                fitnesses = model_forward_api.eval(solutions)
            else:
                fitnesses = [model_forward_api.eval(x) for x in solutions]
            local_es.tell(solutions, fitnesses)
            # es.logger.add()  # write data to disc to be plotted
            # es.disp()
        end_time = time.time()
        print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
        # print('Evaluate on test data...')
        # test_acc = model_forward_api.eval(test_data=test_data)
        # print('Test acc: {}'.format(round(test_acc, 4)))

        client_prompt_dict[idx].append(copy.deepcopy(local_es.mean))

        # Generate solutions uploaded to the server
        solutions = local_es.ask()
        if parallel:
            fitnesses = model_forward_api.eval(solutions)
        else:
            fitnesses = [model_forward_api.eval(x) for x in solutions]
        global_solutions.append(solutions)
        global_fitnesses.append(fitnesses)

        client_fitnesses_dict[idx].append(copy.deepcopy(fitnesses))

        # fitlog.finish()
        client_api_setting_list[idx] = model_forward_api.client_record()

        client_sigmas[idx] = local_sigmas
        print(f'client sigma: {local_sigmas}')



    global_solutions = np.concatenate(global_solutions, axis=0)
    global_fitnesses = np.concatenate(global_fitnesses)

    # caculate global sigma
    client_sigma_list = []
    for ck in client_sigmas.keys():
        client_sigma_list.append(np.sum(client_sigmas[ck]))
    global_sigma = np.sqrt(np.sum(np.array(client_sigma_list)**2)/m)

    
    if schedule_start is True:
        global_es.sigma = global_sigma

    print('Check sigma before: {}'.format(global_es.sigma))

    global_es.ask()
    global_es.tell(global_solutions, global_fitnesses)

    server_prompts.append(copy.deepcopy(global_es.mean))
    print('Check sigma after: {}'.format(global_es.sigma))

    # global_es.sigma = global_sigma/args.local_iter

    print('Global es evaluate on test data...')
    global_api_setting['best_prompt'] = global_es.mean
    model_forward_api.load_client_record(global_api_setting)
    test_acc = model_forward_api.eval(test_data=test_data)
    print('Global test acc: {}'.format(round(test_acc, 4)))

    if schedule_start is True:
        global_es.sigma = global_es.sigma/args.local_iter

    print('Check sigma local: {}'.format(global_es.sigma))

    if global_es.sigma > 6:
        schedule_start = True
        print("Sigma scheduler starts...")

    global_api_setting = model_forward_api.client_record()

    save_prompt_dict = {}
    save_fitness_dict = {}
    for c in range(args.num_users):
        save_prompt_dict[c] = np.vstack(client_prompt_dict[c])
        save_fitness_dict[c] = np.vstack(client_fitnesses_dict[c])
    save_prompt_dict[-1] = np.vstack(server_prompts)

    f= open(f'results/{args.task_name}/schedule_sigma/prompt/fedbbt_prompt_iid{args.iid}_alpha{args.alpha_dir}_popsize{args.local_popsize}.pkl', 'wb+')
    pickle.dump(save_prompt_dict, f)
    f.close()

    f= open(f'results/{args.task_name}/schedule_sigma/prompt/fedbbt_fitness_iid{args.iid}_alpha{args.alpha_dir}_popsize{args.local_popsize}.pkl', 'wb+')
    pickle.dump(save_fitness_dict, f)
    f.close()

