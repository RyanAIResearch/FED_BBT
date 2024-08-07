import os
import copy

import torch

import numpy as np
from fastNLP import Tester, DataSet
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    BartConfig,
    BartTokenizer,
    T5Config,
    T5Tokenizer,
    GPT2Config,
    GPT2Tokenizer,
    BartConfig as CPTConfig,
    LlamaConfig,
    AutoTokenizer
)
from models.modeling_roberta import RobertaForMaskedLM
from models.modeling_bart import BartForConditionalGeneration
from models.modeling_t5 import T5ForConditionalGeneration
from models.modeling_gpt2 import GPT2LMHeadModel
from models.modeling_bert import BertForMaskedLM
from models.modeling_electra import ElectraForMaskedLM
from models.modeling_cpt import CPTForMaskedLM
from models.modeling_llama import LlamaForSequenceClassification, LlamaForCausalLM
from utils import hinge_loss
from sklearn.metrics import f1_score










class LMBackwardAPI:
    def __init__(self, args, train_data=None, dev_data=None, init_prompt_path=None, baseAPI=True):
        model_name = args.model_name
        if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            from metrics.metrics_t5 import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            from metrics.metrics_gpt import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
        elif model_name in ['fnlp/cpt-large']:
            from metrics.metrics_cpt import ChnSentMetric, AmazonMetric, THUCNewsMetric, BQMetric, CMNLIMetric, CCPMMetric, TNewsMetric, OCNLIMetric, LCQMCMetric, C3Metric
        elif model_name in ['llama2']:
            if args.llama_causal:
                from metrics.metrics_llama_causal import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
            else:
                from metrics.metrics_llama import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
        else:
            from metrics.metrics import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric
        task_name = args.task_name
        if task_name in ['sst2', 'yelpp', 'rte', 'mrpc', 'chnsent', 'lcqmc', 'bq']:
            self.num_labels = 2
        elif task_name in ['snli', 'cmnli', 'ocnli']:
            self.num_labels = 3
        elif task_name in ['agnews', 'ccpm', 'c3']:
            self.num_labels = 4
        elif task_name in ['amazon']:
            self.num_labels = 5
        elif task_name in ['thucnews']:
            self.num_labels = 10
        elif task_name in ['dbpedia', 'tnews']:
            self.num_labels = 14
        else:
            raise ValueError
        n_prompt_tokens = args.n_prompt_tokens
        intrinsic_dim = args.intrinsic_dim

        sigma = args.sigma
        alpha = args.alpha
        self.args = args

        device = args.device
        random_proj = args.random_proj
        loss_type = args.loss_type
        print_every = args.print_every
        eval_every = args.eval_every
        cat_or_add = args.cat_or_add
        
        inference_framework = args.inference_framework
        onnx_model_path = args.onnx_model_path

        self.model_name = args.model_name
        self.parallel = args.parallel
        self.n_prompt_tokens = args.n_prompt_tokens
        self.batch_size = args.batch_size
        self.device =args.device

        if inference_framework not in ['pt', 'ort']:
            raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
        if inference_framework == 'ort':
            assert onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
            assert os.path.exists(onnx_model_path), f'In valid onnx model path `{onnx_model_path}`'


        self.train_data = train_data
        self.dev_data = dev_data
        self.train_data_aux = None
        if model_name in ['roberta-base', 'roberta-large']:
            self.config = RobertaConfig.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
                inference_framework=inference_framework,
                onnx_model_path=onnx_model_path,
            )
            self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
            self.config = BertConfig.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['google/electra-base-generator', 'google/electra-large-generator']:
            self.config = ElectraConfig.from_pretrained(model_name)
            self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
            self.model = ElectraForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['facebook/bart-base', 'facebook/bart-large']:
            self.config = BartConfig.from_pretrained(model_name)
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            self.config = T5Config.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            self.config = GPT2Config.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['fnlp/cpt-large']:
            self.config = CPTConfig.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = CPTForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['llama2']:
            self.config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
            self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.config.pad_token_id = self.tokenizer.pad_token_id
            self.config.num_labels = self.num_labels
            if args.llama_causal:
                self.model = LlamaForCausalLM.from_pretrained(
                    'meta-llama/Llama-2-7b-hf',
                    config=self.config,
                    n_prompt_tokens=n_prompt_tokens,
                )
            else:
                self.model = LlamaForSequenceClassification.from_pretrained(
                    'meta-llama/Llama-2-7b-hf',
                    config=self.config,
                    n_prompt_tokens=n_prompt_tokens,
                )
        else:
            raise NotImplementedError
        if inference_framework == 'ort':
            self.model.roberta = None
        if cat_or_add == 'cat':
            self.model.set_concat_prompt(True)
            if init_prompt_path is not None:
                print('Initialize prompt embedding from {}'.format(init_prompt_path))
                self.init_prompt = torch.load(init_prompt_path).weight.cpu().reshape(-1)
            else:
                print('Initial prompt embedding not found. Initialize to zero embedding.')
                self.init_prompt = torch.zeros(n_prompt_tokens * self.config.hidden_size)
            print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        else:
            # self.model.set_concat_prompt(False)
            self.init_prompt = None
            
        if args.init_score_path is not None:
            if args.llama_causal:
                raise ValueError("You cannot initilize a score layer for a causal model")
            self.model.score.load_state_dict(torch.load(args.init_score_path))
        
        self.model.to(device)
        self.model.eval()
        # self.linear = torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False).to(device)
        # if random_proj == 'normal':
        #     # calculate std for normal distribution
        #     if model_name in ['roberta-base', 'roberta-large']:
        #         embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
        #     elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
        #         embedding = self.model.bert.get_input_embeddings().weight.clone().cpu()
        #     elif model_name in ['google/electra-base-generator', 'google/electra-large-generator']:
        #         embedding = self.model.electra.get_input_embeddings().weight.clone().cpu()
        #     elif model_name in ['facebook/bart-base', 'facebook/bart-large', 'fnlp/cpt-large']:
        #         embedding = self.model.model.get_input_embeddings().weight.clone().cpu()
        #     elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        #         embedding = self.model.transformer.get_input_embeddings().weight.clone().cpu()
        #     else:  # T5
        #         embedding = self.model.get_input_embeddings().weight.clone().cpu()
        #     # embedding = embedding[1000: 2000]
        #     mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
        #     std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        #     mu = 0.0
        #     std = alpha * std_hat / (np.sqrt(intrinsic_dim) * sigma)
        #     # temp = intrinsic_dim - std_hat * std_hat
        #     # mu = mu_hat / temp
        #     # std = std_hat / np.sqrt(temp)
        #     print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        #     for p in self.linear.parameters():
        #         torch.nn.init.normal_(p, mu, std)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        # self.save_path = save_path
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type
        # if save_path is not None:
        #     os.makedirs(save_path, exist_ok=True)
        if task_name == 'sst2':
            self.metric = SST2Metric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SST2Metric'
        elif task_name == 'agnews':
            self.metric = AGNewsMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'AGNewsMetric'
        elif task_name == 'yelpp':
            self.metric = YelpPMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'YelpPMetric'
        elif task_name == 'dbpedia':
            self.metric = DBPediaMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'DBPediaMetric'
        elif task_name == 'rte':
            self.metric = RTEMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'RTEMetric'
        elif task_name == 'mrpc':
            self.metric = MRPCMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'f1'
            self.metric_name = 'MRPCMetric'
        elif task_name == 'snli':
            self.metric = SNLIMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SNLIMetric'
        elif task_name == 'chnsent':
            self.metric = ChnSentMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'ChnSentMetric'
        elif task_name == 'thucnews':
            self.metric = THUCNewsMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'THUCNewsMetric'
        elif task_name == 'lcqmc':
            self.metric = LCQMCMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'LCQMCMetric'
        elif task_name == 'cmnli':
            self.metric = CMNLIMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'CMNLIMetric'
        elif task_name == 'ocnli':
            self.metric = OCNLIMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'OCNLIMetric'
        elif task_name == 'amazon':
            self.metric = AmazonMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'AmazonMetric'
        elif task_name == 'bq':
            self.metric = BQMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'BQMetric'
        elif task_name == 'ccpm':
            self.metric = CCPMMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'CCPMMetric'
        elif task_name == 'tnews':
            self.metric = TNewsMetric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'TNewsMetric'
        elif task_name == 'c3':
            self.metric = C3Metric(target='labels', pred='logits', tokenizer=self.tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'C3Metric'
        else:
            raise NotImplementedError
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def convert_pred(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        if self.args.model_name not in ['llama2'] or self.args.llama_causal:
            interest_index = list(label_map.keys())
            logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)
        return pred, converted_target

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        if self.args.model_name not in ['llama2'] or self.args.llama_causal:
            interest_index = list(label_map.keys())
            logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum') / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target)
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf
    
    def set_dataset(self, train_data, dev_data, train_data_aux=None):
        self.train_data, self.dev_data = train_data, dev_data
        if train_data_aux is not None:
            self.train_data_aux = train_data_aux

    def load_client_record(self, record):
        self.best_train_perf = record['best_train_perf']
        self.best_dev_perf = record['best_dev_perf']
        self.best_prompt = record['best_prompt']
        self.num_call = record['num_call']

    def client_record(self):
        record = {}
        record['best_train_perf'] = copy.deepcopy(self.best_train_perf)
        record['best_dev_perf'] = copy.deepcopy(self.best_dev_perf)
        record['best_prompt'] = copy.deepcopy(self.best_prompt)
        record['num_call'] = copy.deepcopy(self.num_call)
        return record
    
    def zero_grad(self):
        self.model.zero_grad()

    # def inference(self, model, data):
    #     for k, v in data.items():
    #         data[k] = v.to(self.device)
    #     with torch.no_grad():
    #         if self.model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    #             logits = self.model(
    #                 input_ids=data['input_ids'],
    #                 attention_mask=data['attention_mask'],
    #                 decoder_input_ids=data['decoder_input_ids'],
    #                 decoder_attention_mask=data['decoder_attention_mask'],
    #             )['logits']
    #         elif self.model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    #             logits = self.model(
    #                 input_ids=data['input_ids'],
    #                 attention_mask=data['attention_mask'],
    #             )['logits']
    #         else:
    #             logits = self.model(
    #                 input_ids=data['input_ids'],
    #                 attention_mask=data['attention_mask'],
    #                 mask_pos=data['mask_pos'],
    #             )['logits']

    #     target = data['labels']
    #     label_map = self.metric.label_map

    #     converted_target = target.clone()
    #     for key, val in label_map.items():
    #         converted_target[target == key] = val
    #     interest_index = list(label_map.keys())
    #     logits = logits[:, interest_index]
    #     pred = logits.argmax(dim=-1)
    #     return pred, converted_target

    def eval(self, prompt_embedding, train_data=None, train_label=None, test_data=None, return_pred=False):
        self.num_call += 1
        if test_data is None:
            bsz_dev = len(self.dev_data['input_ids'])
            bsz_train = len(self.train_data['input_ids'])
            bsz = bsz_train if bsz_train>bsz_dev else bsz_dev
        else:
            bsz = self.batch_size  # for test data

        if isinstance(prompt_embedding, torch.Tensor):
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt
            prompt_embedding = prompt_embedding.reshape(self.n_prompt_tokens, -1).repeat(bsz, 1, 1)
        elif prompt_embedding is not None:
            raise ValueError(
                f'[Prompt Embedding] Only support [Tensor, None], got `{type(prompt_embedding)}` instead.'
            )
        
        self.model.set_prompt_embedding(prompt_embedding)

        if return_pred is True:
            if self.parallel:  # if we have multiple queries, use the one that achieves minimal loss
                self.model.set_prompt_embedding(prompt_embedding)
            for k, v in self.dev_data.items():
                self.dev_data[k] = v.to(self.device)
            with torch.no_grad():
                if self.model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                    logits = self.model(
                        input_ids=self.dev_data['input_ids'],
                        attention_mask=self.dev_data['attention_mask'],
                        decoder_input_ids=self.dev_data['decoder_input_ids'],
                        decoder_attention_mask=self.dev_data['decoder_attention_mask'],
                    )['logits']
                elif self.model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama2']:
                    logits = self.model(
                        input_ids=self.dev_data['input_ids'],
                        attention_mask=self.dev_data['attention_mask'],
                    )['logits']
                else:
                    logits = self.model(
                        input_ids=self.dev_data['input_ids'],
                        attention_mask=self.dev_data['attention_mask'],
                        mask_pos=self.dev_data['mask_pos'],
                    )['logits']
            pred, labels = self.convert_pred(logits, self.dev_data['labels'])
            return pred, labels

        if isinstance(test_data, DataSet):
            if prompt_embedding is not None:
                if prompt_embedding.shape[0] > bsz:
                    raise ValueError('Provide a single prompt embedding for testing.')
            
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric, batch_size=self.batch_size,
                                 num_workers=1, device=self.device, use_tqdm=False)
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            for k, v in train_data.items():
                train_data[k] = v.to(self.device)
            for k, v in train_label.items():
                train_label[k] = v.to(self.device)
            if self.model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                logits = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    decoder_input_ids=train_data['decoder_input_ids'],
                    decoder_attention_mask=train_data['decoder_attention_mask'],
                )['logits']
            elif self.model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'llama2']:
                logits = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                )['logits']
            else:
                logits = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    mask_pos=train_data['mask_pos'],
                )['logits']


            loss, perf = self.calc_metric(logits, train_label['labels'])
            # fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            # fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            return loss, perf