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

    def forward(self, input_ids, attention_mask):
        return torch.sum([m(input_ids, attention_mask) for m in self.models]) / self.n

    def predict_sentiment(self, input_ids, attention_mask):
        return self.models[SST].predict_sentiment(input_ids, attention_mask)

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        return self.models[PARA].predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        return self.models[STS].predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
    
    def cos_sim_loss(self, sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels):
        return self.models[STS].cos_sim_loss(sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels)
    
    def multiple_negatives_ranking_loss(self, sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels):
        return self.models[STS].multiple_negatives_ranking_loss(sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels)

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
        
        sst_hidden_size, para_hidden_size, sts_hidden_size = \
            config.sst_hidden_size, config.para_hidden_size, config.sts_hidden_size
        sst_layers, para_layers, sts_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        if config.num_sst_layers > 1:
            sst_layers.append(FF(BERT_HIDDEN_SIZE, sst_hidden_size, INPUT_DROP))
            sst_layers.extend([FF(sst_hidden_size, sst_hidden_size, HIDDEN_DROP) for _ in range(config.num_sst_layers - 2)])
            sst_layers.append(FF(sst_hidden_size, N_SENTIMENT_CLASSES, OUTPUT_DROP))
        else:
            sst_layers.append(FF(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES))

        if config.num_para_layers > 1:
            para_layers.append(FF(2*BERT_HIDDEN_SIZE, para_hidden_size, INPUT_DROP))
            para_layers.extend([FF(para_hidden_size, para_hidden_size, HIDDEN_DROP) for _ in range(config.num_para_layers - 2)])
            para_layers.append(FF(para_hidden_size, 1))
        else:
            para_layers.append(FF(2*BERT_HIDDEN_SIZE, 1, OUTPUT_DROP))
        
        if config.num_sts_layers > 1:
            sts_layers.append(FF(2*BERT_HIDDEN_SIZE, sts_hidden_size, INPUT_DROP))
            sts_layers.extend([FF(sts_hidden_size, sts_hidden_size, HIDDEN_DROP) for _ in range(config.num_sts_layers - 2)])
            sts_layers.append(FF(sts_hidden_size, 1, OUTPUT_DROP))
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

    '''
    Returns negative ranking loss of pairs of equivalent sentences

    1) Filters out all examples of sentence pairs that aren't equivalent
    2) Evaluates cosine similarity for each sentence 1 and all possible sentence 2
    3) Returns cross entropy loss, with correct label being the diagonal, as well as number of examples considered
    
    '''
    def multiple_negatives_ranking_loss(self, sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels):
        #1) Filters out all examples of sentence pairs that aren't equivalent
        mask = torch.where(sts_labels >= 3.5, True, False)
        masked_ids_1 = sts_ids_1[mask,:]
        masked_ids_2 = sts_ids_2[mask,:]
        masked_mask_1 = sts_mask_1[mask,:]
        masked_mask_2 = sts_mask_2[mask,:]

        #2) Evaluates cosine similarity for each sentence 1 and all possible sentence 2
        emb_1 = self.bert(masked_ids_1, masked_mask_1)['last_hidden_state'][:,0,:]
        emb_2 = self.bert(masked_ids_2, masked_mask_2)['last_hidden_state'][:,0,:]
        similarity_matrix = F.cosine_similarity(emb_1, emb_2, dim=1)

        # 3) Returns cross entropy loss, with correct label being the diagonal
        n = len(masked_ids_1)
        labels = torch.arange(n).to(similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels, reduction = 'sum')
        return loss, n
    
    '''
    Cosine similarity loss function for similarity task.
    '''
    def cos_sim_loss(self, sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels):
        mask = torch.where((sts_labels < 1.0) | (sts_labels >= 3.5), True, False)
        cos_sim_labels = torch.where(sts_labels[mask] < 1.0, -1, 1) # -1 marks unrelated sentences, 1 equivalent sentences
        cos_sim_ids_1 = sts_ids_1[mask,:]
        cos_sim_ids_2 = sts_ids_2[mask,:]
        cos_sim_mask_1 = sts_mask_1[mask,:]
        cos_sim_mask_2 = sts_mask_2[mask,:]
        cos_sim_emb_1 = self.bert(cos_sim_ids_1, cos_sim_mask_1)['last_hidden_state'][:,0,:]
        cos_sim_emb_2 = self.bert(cos_sim_ids_2, cos_sim_mask_2)['last_hidden_state'][:,0,:]
        cos_loss = F.cosine_embedding_loss(cos_sim_emb_1, cos_sim_emb_2, cos_sim_labels, reduction='sum')
        n = len(cos_sim_ids_1)
        return cos_loss, n