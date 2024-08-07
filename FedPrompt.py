import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

from LMBackwardAPI import LMBackwardAPI
from data_process import data_processor, construct_true_few_shot_data, split_data
from models.prompt_encoder import PromptEncoder
import transformers

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

print(len(train_data[0]['input_ids']))
print(len(train_data[0]['attention_mask']))

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


# initialize prompt embeddings

prefix_tokens = torch.arange(n_prompt_tokens).long().to(args.device)
if args.p_tuning == 1:
    global_prefix_encoder = PromptEncoder((args.n_prompt_tokens, 0, 0), model_forward_api.config.hidden_size, data_processor.tokenizer, args.device, args)
else:
    global_prefix_encoder = torch.nn.Embedding(n_prompt_tokens, model_forward_api.config.hidden_size).to(args.device)
    torch.nn.init.xavier_uniform_(global_prefix_encoder.weight, gain=1)

init_prompt_embedding = global_prefix_encoder(prefix_tokens)

test_acc = model_forward_api.eval(prompt_embedding=init_prompt_embedding, test_data=test_data)
print('Global test acc: {}'.format(round(test_acc, 4)))
global_api_setting = model_forward_api.client_record()



for e in range(args.epochs):

    print(f"Global epoch {e}...")
    m = max(int(args.frac * args.num_users), 1)
    #sample users for training
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    local_weights = []
    local_scores = []
    for idx in idxs_users:

        local_prefix_encoder = copy.deepcopy(global_prefix_encoder)
        if args.p_tuning == 1:
            if args.model_name in ['llama2']:
                optimizer = torch.optim.Adam(params=local_prefix_encoder.parameters(), lr=1e-4)
            else:
                optimizer = torch.optim.Adam(params=local_prefix_encoder.parameters(), lr=1e-3)
        else:
            optimizer = torch.optim.Adam(params=local_prefix_encoder.parameters(), lr=1e-2)

        if args.model_name in ['llama2'] and not args.llama_causal:
            para_list = []
            for name,para in model_forward_api.model.named_parameters():
                if 'score' in name:
                    para_list.append(para)
            optimizer_score = transformers.AdamW(para_list, lr=5e-5)
        
        model_forward_api.load_client_record(client_api_setting_list[idx])
        # initialize local data
        train_sample_idxs, dev_sample_idxs = user_dict_train[idx], user_dict_dev[idx]
        print(f"Client {idx} execute local training on {len(train_sample_idxs)} samples...")

        # if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
        #     local_train_data = {
        #         'input_ids': torch.tensor(train_data['input_ids'].get(train_sample_idxs)),
        #         'attention_mask': torch.tensor(train_data['attention_mask'].get(train_sample_idxs)),
        #         'decoder_input_ids': torch.tensor(train_data['decoder_input_ids'].get(train_sample_idxs)),
        #         'decoder_attention_mask': torch.tensor(train_data['decoder_attention_mask'].get(train_sample_idxs)),
        #         'labels': torch.tensor(train_data['labels'].get(train_sample_idxs)),
        #     }
        #     local_dev_data = {
        #         'input_ids': torch.tensor(dev_data['input_ids'].get(dev_sample_idxs)),
        #         'attention_mask': torch.tensor(dev_data['attention_mask'].get(dev_sample_idxs)),
        #         'decoder_input_ids': torch.tensor(dev_data['decoder_input_ids'].get(dev_sample_idxs)),
        #         'decoder_attention_mask': torch.tensor(dev_data['decoder_attention_mask'].get(dev_sample_idxs)),
        #         'labels': torch.tensor(dev_data['labels'].get(dev_sample_idxs)),
        #     }
        # elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        #     local_train_data = {
        #         'input_ids': torch.tensor(train_data['input_ids'].get(train_sample_idxs)),
        #         'attention_mask': torch.tensor(train_data['attention_mask'].get(train_sample_idxs)),
        #         'labels': torch.tensor(train_data['labels'].get(train_sample_idxs)),
        #     }
        #     local_dev_data = {
        #         'input_ids': torch.tensor(dev_data['input_ids'].get(dev_sample_idxs)),
        #         'attention_mask': torch.tensor(dev_data['attention_mask'].get(dev_sample_idxs)),
        #         'labels': torch.tensor(dev_data['labels'].get(dev_sample_idxs)),
        #     }
        # else:
        #     local_train_data = {
        #         'input_ids': torch.tensor(train_data['input_ids'].get(train_sample_idxs)),
        #         'attention_mask': torch.tensor(train_data['attention_mask'].get(train_sample_idxs)),
        #         'mask_pos': torch.tensor(train_data['mask_pos'].get(train_sample_idxs)),
        #         'labels': torch.tensor(train_data['labels'].get(train_sample_idxs)),
        #     }
        #     local_dev_data = {
        #         'input_ids': torch.tensor(dev_data['input_ids'].get(dev_sample_idxs)),
        #         'attention_mask': torch.tensor(dev_data['attention_mask'].get(dev_sample_idxs)),
        #         'mask_pos': torch.tensor(dev_data['mask_pos'].get(dev_sample_idxs)),
        #         'labels': torch.tensor(dev_data['labels'].get(dev_sample_idxs)),
        #     }


        # local_train_data = DataSet(local_train_data)
        # local_dev_data = DataSet(local_dev_data)


        local_train_data = train_data[np.array(list(train_sample_idxs)).tolist()]
        local_dev_data = dev_data[np.array(list(dev_sample_idxs)).tolist()]

        dataloader_train = DataSetIter(local_train_data, batch_size=args.batch_size, sampler=SequentialSampler())


        model_forward_api.set_dataset(local_train_data, local_dev_data)

        # opt = cma.CMAOptions()
        start_time = time.time()

        for le in range(args.local_epochs):
            for batch_x, batch_y in dataloader_train:
                optimizer.zero_grad()
                model_forward_api.zero_grad()
                local_prompt = local_prefix_encoder(prefix_tokens)
                loss, _ = model_forward_api.eval(local_prompt, train_data=batch_x, train_label=batch_y)
                loss.backward()
                optimizer.step()
                if args.model_name in ['llama2'] and not args.llama_causal:
                    optimizer_score.step()
                    optimizer_score.zero_grad()

            print('Local loss @ local epoch {}: {}'.format(le, loss.item()))
            
            
            # es.logger.add()  # write data to disc to be plotted
            # es.disp()
        local_prompt = local_prefix_encoder(prefix_tokens)
        test_acc = model_forward_api.eval(prompt_embedding=local_prompt, test_data=test_data)
        print('Local test acc @ epoch {}: {}'.format(e, round(test_acc, 4)))
        end_time = time.time()

        local_weights.append(copy.deepcopy(local_prefix_encoder.state_dict()))
        if args.model_name in ['llama2'] and not args.llama_causal:
            local_scores.append(copy.deepcopy(model_forward_api.model.score_state_dict()))

        client_api_setting_list[idx] = model_forward_api.client_record()

    global_weights = average_weights(local_weights)
    global_prefix_encoder.load_state_dict(global_weights)
    if args.model_name in ['llama2'] and not args.llama_causal:
        model_forward_api.model.load_score_state_dict(average_weights(local_scores))


   
    print('Global evaluate on test data...')
    global_prompt_embedding = global_prefix_encoder(prefix_tokens)
    test_acc = model_forward_api.eval(prompt_embedding=global_prompt_embedding, test_data=test_data)
    print('Global test acc : {}'.format(round(test_acc, 4)))
    global_api_setting = model_forward_api.client_record()
    print('Global prompt norm: {}'.format(np.linalg.norm(global_prompt_embedding.cpu().detach().numpy())))

    # f= open(f'results/{args.task_name}/prompt/fedavg_prompt_iid{args.iid}_alpha{args.alpha_dir}_popsize{args.local_popsize}.pkl', 'wb+')
    # pickle.dump(client_prompt_dict, f)
    # f.close()

