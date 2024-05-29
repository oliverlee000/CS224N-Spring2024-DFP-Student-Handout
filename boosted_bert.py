import random, numpy as np, argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from torch.utils.data import DataLoader
from optimizer import AdamW
from bert import BertModel

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
DROPOUT_PROB = 0.1

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

class BoostedBERT(nn.Module):
    def __init__(self, config):
        super(BoostedBERT, self).__init__()
        self.n = NUM_TASKS
        self.models = nn.ModuleList([MultitaskBERT(config) for _ in range(NUM_TASKS)])
        print(self.models[0].parameters())

    def forward(self, input_ids, attention_mask):
        return torch.sum([m(input_ids, attention_mask) for m in self.models]) / self.n

    def predict_sentiment(self, input_ids, attention_mask):
        return self.models[SST].predict_sentiment(input_ids, attention_mask)

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        return self.models[PARA].predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        return self.models[STS].predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
    
    def multiple_negatives_ranking_loss(self, embeddings, batch_size):
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        labels = torch.arange(batch_size).to(similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
DROPOUT_PROB = 0.1

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
        self.sst_layers = nn.ModuleList([FF(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE) for _ in range(config.num_sst_layers - 1)])
        self.sst_layers.append(FF(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES))
        self.para_layers = nn.ModuleList([FF(2*BERT_HIDDEN_SIZE, 2*BERT_HIDDEN_SIZE) for _ in range(config.num_para_layers - 1)])
        self.para_layers.append(FF(2*BERT_HIDDEN_SIZE, 1))
        self.sts_layers = nn.ModuleList([FF(2*BERT_HIDDEN_SIZE, 2*BERT_HIDDEN_SIZE, relu=True) for _ in range(config.num_sts_layers - 1)])
        self.sts_layers.append(FF(2*BERT_HIDDEN_SIZE, 1))

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
        if len(self.para_layers) == 0:
            return torch.dot(output1, output2)
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

'''
Consists exclusively of a feed forward layer
'''
class FF(nn.Module):
    def __init__(self, hidden_size, output_size, relu=False):
        super().__init__()
        # Feed forward.
        self.dropout = nn.Dropout(DROPOUT_PROB)
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

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")