import random, numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from torch.utils.data import DataLoader
from bert import BertModel

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

# Dropout probabilities
INPUT_DROP = 0.1
HIDDEN_DROP = 0.4
OUTPUT_DROP = 0.0

NUM_TASKS = 3
SST = 0
PARA = 1
STS = 2

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

'''
Ensemble of three BERT models, each responsible for one task.'''
class BoostedBERT(nn.Module):
    def __init__(self, config):
        super(BoostedBERT, self).__init__()
        self.n = NUM_TASKS
        self.models = nn.ModuleList([MultitaskBERT(config) for _ in range(NUM_TASKS)])
        print([name for name, p in self.models[0].named_parameters()])

    def forward(self, input_ids, attention_mask):
        return torch.sum([m(input_ids, attention_mask) for m in self.models]) / self.n

    def predict_sentiment(self, input_ids, attention_mask):
        return self.models[SST].predict_sentiment(input_ids, attention_mask)

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        return self.models[PARA].predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        return self.models[STS].predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
    
    def cosine_similarity_fine_tuning(self, output1, output2):
        return self.models[STS].cosine_similarity_fine_tuning(self, output1, output2)

'''
Consists exclusively of a feed forward layer
'''
class FF(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob=HIDDEN_DROP, relu=False):
        super().__init__()
        # Feed forward.
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, output_size)
        self.af = F.gelu
        if relu:
            self.af = F.relu


    def forward(self, hidden_states, activation=True):
        """
        Put elements in feed forward.
        Feed forward consists of:
        1. a dropout layer,
        2. a linear layer, and
        3. an activation function.

        If activation = True, use activation
        """
        # TODO
        hidden_states = self.dropout(hidden_states)
        output = self.dense(hidden_states)
        if activation:
            output = self.af(output)
        return output

'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''
class MultitaskBERT(nn.Module):
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        
        hidden_size = config.hidden_size
        sst_layers, para_layers, sts_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        if config.num_sst_layers > 1:
            sst_layers.append(FF(BERT_HIDDEN_SIZE, hidden_size, INPUT_DROP))
            sst_layers.extend([FF(hidden_size, hidden_size, HIDDEN_DROP) for _ in range(config.num_sst_layers - 2)])
            sst_layers.append(FF(hidden_size, N_SENTIMENT_CLASSES, OUTPUT_DROP))
        else:
            sst_layers.append(FF(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES))

        if config.num_para_layers > 1:
            para_layers.append(FF(2*BERT_HIDDEN_SIZE, hidden_size, INPUT_DROP))
            para_layers.extend([FF(hidden_size, hidden_size, HIDDEN_DROP) for _ in range(config.num_para_layers - 2)])
            para_layers.append(FF(hidden_size, 1))
        else:
            para_layers.append(FF(2*BERT_HIDDEN_SIZE, 1, OUTPUT_DROP))
        
        if config.num_sts_layers > 1:
            sts_layers.append(FF(2*BERT_HIDDEN_SIZE, hidden_size, INPUT_DROP))
            sts_layers.extend([FF(hidden_size, hidden_size, HIDDEN_DROP) for _ in range(config.num_sts_layers - 2)])
            sts_layers.append(FF(hidden_size, 1, OUTPUT_DROP))
        else:
            sts_layers.append(FF(2*BERT_HIDDEN_SIZE, 1, OUTPUT_DROP))
        self.sst_layers, self.para_layers, self.sts_layers = sst_layers, para_layers, sts_layers

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        return output['last_hidden_state']

    def predict_sentiment(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        embeds = output['last_hidden_state'][:,0,:]
        for i, layer_module in enumerate(self.sst_layers[:-1]):
            embeds = layer_module(embeds, activation=True)
        output = self.sst_layers[-1](embeds, activation=False)
        return output

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output1 = self.bert(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        output2 = self.bert(input_ids_2, attention_mask_2)['last_hidden_state'][:,0,:]
        embeds = torch.cat((output1, output2), 1)
        for i, layer_module in enumerate(self.para_layers[:-1]):
            embeds = layer_module(embeds, activation=True)
        output_agr = self.para_layers[-1](embeds, activation=False)
        return output_agr

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        output1 = self.bert(input_ids_1, attention_mask_1)
        output2 = self.bert(input_ids_2, attention_mask_2)
        embeds = torch.cat((output1['pooler_output'], output2['pooler_output']), 1)
        for i, layer_module in enumerate(self.sts_layers[:-1]):
            embeds = layer_module(embeds, activation=True)
        output_agr = self.sts_layers[-1](embeds, activation=False)
        return output_agr

    def multiple_negatives_ranking_loss(self, embeddings, batch_size):
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        labels = torch.arange(batch_size).to(similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def cosine_similarity_fine_tuning(self, output1, output2):
        cosine_sim = F.cosine_similarity(output1, output2, dim=-1)
        return cosine_sim