import random, numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from torch.utils.data import DataLoader
from bert import BertModel
from lora import LoraBertModel

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
    
    def multiple_negatives_ranking_loss(self, sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2):
        return self.models[STS].multiple_negatives_ranking_loss(sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2)

'''
Neural net consisting of feed forward layers'''
class NN(nn.Module):
    def __init__(self, n, input_size, hidden_size, output_size, input_drop=INPUT_DROP, hidden_drop=HIDDEN_DROP, output_drop=OUTPUT_DROP, relu=False):
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList()
        if n > 1:
            self.layers.append(FF(input_size, hidden_size, input_drop, relu))
            self.layers.extend([FF(hidden_size, hidden_size, hidden_drop, relu) for _ in range(n - 2)])
            self.layers.append(FF(hidden_size, output_size, output_drop, relu))
        elif n == 1:
            self.layers.append(FF(input_size, output_size, output_drop, relu))
    
    def forward(self, input):
        if self.n == 0:
            return input
        for _, layer_module in enumerate(self.layers[:-1]):
            input = layer_module(input, activation=True)
        output = self.layers[-1](input, activation=False)
        return output

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
        # Create LoraBertModel if lora flag is on
        if config.lora == 'y':
            self.bert = LoraBertModel.from_pretrained('bert-base-uncased', lora_size=config.lora_size)
            # Training parameters are by default marked true, so turn off in fine tune mode is last linear layer
            for param in self.bert.parameters():
                if config.fine_tune_mode == 'last-linear-layer':
                    param.requires_grad = False

        sst_hidden_size, para_hidden_size, sts_hidden_size = \
            config.sst_hidden_size, config.para_hidden_size, config.sts_hidden_size
        
        self.sst_layers, self.para_layers, self.sts_layers = NN(config.num_sst_layers, BERT_HIDDEN_SIZE, sst_hidden_size, N_SENTIMENT_CLASSES), \
            NN(config.num_para_layers, BERT_HIDDEN_SIZE, para_hidden_size, para_hidden_size), NN(config.num_sts_layers, BERT_HIDDEN_SIZE, sts_hidden_size, sts_hidden_size)
        

        # If concat, then concatenate input embeddings and push into feed forward; else take dot product
        self.para_concat, self.sts_concat = config.para_concat, config.sts_concat

        if self.para_concat == 'y':
            self.para_layers = NN(config.num_para_layers, 2*BERT_HIDDEN_SIZE, para_hidden_size, 1)

        if self.sts_concat == 'y':
            self.sts_layers = NN(config.num_sts_layers, 2*BERT_HIDDEN_SIZE, sts_hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        return output['last_hidden_state']

    def predict_sentiment(self, input_ids, attention_mask):
        embed = self.bert(input_ids, attention_mask)['last_hidden_state'][:,0,:]
        output = self.sst_layers(embed)
        return output

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        embed_1 = self.bert(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        embed_2 = self.bert(input_ids_2, attention_mask_2)['last_hidden_state'][:,0,:]
        if self.para_concat == 'y':
            # Concanate embeddings together, then return NN output of concatenation
            embeds = torch.cat((embed_1, embed_2), 1)
            output = self.para_layers(embeds)
            return output
        else:
            output_1 = self.para_layers(embed_1)
            output_2 = self.para_layers(embed_2)
            output = F.cosine_similarity(output_1, output_2).view(-1, 1)
            return output

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        embed_1 = self.bert(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        embed_2 = self.bert(input_ids_2, attention_mask_2)['last_hidden_state'][:,0,:]
        if self.sts_concat == 'y':
            # Concanate embeddings together, then return NN output of concatenation
            embeds = torch.cat((embed_1, embed_2), 1)
            output = self.sts_layers(embeds)
            return output
        else:
            output_1 = self.sts_layers(embed_1)
            output_2 = self.sts_layers(embed_2)
            output = F.cosine_similarity(output_1, output_2).view(-1, 1)
            return output
        
    def predict_paraphrase_with_emb(self, emb_1, emb_2):
        if self.para_concat == 'y':
            embeds = torch.cat((emb_1, emb_2), 1)
            output = self.para_layers(embeds)
            return output
        else:
            output_1 = self.para_layers(emb_1)
            output_2 = self.para_layers(emb_2)
            output = F.cosine_similarity(output_1, output_2, dim=-1).view(-1, 1)
            return output


    '''
    Returns negative ranking loss of pairs of equivalent sentences.
    We take the cross similarity of each sentences in sts_ids_1 with all sentences in sts_ids_2, optimizing so that
    all pairs of sentences which are labeled as equivalent (score 3.5) or higher, get high cosine similarity scores

    1) Filters out all examples of sentence pairs that aren't equivalent
    2) Evaluates cosine similarity for each sentence 1 and all possible sentence 2
    3) Returns cross entropy loss, with correct label being the diagonal, as well as number of examples considered   
    '''
    def multiple_negatives_ranking_loss(self, sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2):
        #1) Evaluates cosine similarity for each sentence 1 and all possible sentence 2
        emb_1 = self.bert(sts_ids_1, sts_mask_1)['last_hidden_state'][:,0,:]
        emb_2 = self.bert(sts_ids_2, sts_mask_2)['last_hidden_state'][:,0,:]

        similarities = emb_1 @ emb_2.transpose(-1, -2) \
            / torch.linalg.vector_norm(emb_1, dim = 1) / torch.linalg.vector_norm(emb_2.transpose(-1, -2), dim = 0)
        # 3) Returns cross entropy loss, with correct label being the diagonal
        labels = torch.arange(len(sts_ids_1)).to(similarities.device)
        loss = F.cross_entropy(similarities, labels, reduction ='sum')
        return loss
    

    #Cosine similarity loss function for similarity task.

    def cos_sim_loss(self, sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels):
        cos_sim_emb_1 = self.bert(sts_ids_1, sts_mask_1)['last_hidden_state'][:,0,:]
        cos_sim_emb_2 = self.bert(sts_ids_2, sts_mask_2)['last_hidden_state'][:,0,:]
        cos_loss = F.cosine_embedding_loss(cos_sim_emb_1, cos_sim_emb_2, sts_labels, reduction='mean')
        return cos_loss

    def predict_similarity_with_emb(self, emb_1, emb_2):
        if self.sts_concat == 'y':
            embeds = torch.cat((emb_1, emb_2), 1)
            output = self.sts_layers(embeds)
            return output
        else:
            output_1 = self.sts_layers(emb_1)
            output_2 = self.sts_layers(emb_2)
            output = F.cosine_similarity(output_1, output_2, dim=-1).view(-1, 1)
            return output