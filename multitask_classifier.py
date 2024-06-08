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

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from models import BoostedBERT, MultitaskBERT

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    load_sts_data,
    load_cos_sim_data,
    load_neg_rankings_data
)

from evaluation_single import model_eval_para, model_eval_sts, model_eval_test_sst, model_eval_test_para, model_eval_test_sts

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

BERT_HIDDEN_SIZE = 768
FINE_TUNING_DOWNWEIGHT = 1 # downweights cosine similarity and negative ranking loss finetuning

#dimIn = k
#dimOut = d
class LoRADoRA(nn.Module):
    def __init__(self, dimIn, dimOut, rank=4, bias=None, weight=None):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if bias != None:
            self.bias = nn.Parameter(bias, requires_grad=False).to(device)
        else:
            self.bias = nn.zeros(dimOut)
            self.bias = nn.Parameter(self.bias, requires_grad=False).to(device)
        if weight != None:
            self.weight = nn.Parameter(weight, requires_grad=False).to(device)
        else:
            self.weight = nn.zeros(dimOut, dimIn)
            self.weight = nn.Parameter(self.weight, requires_grad=False).to(device)
        
        #calculate m vector using description in handout
        self.mVector = self.weight ** 2
        self.mVector = torch.sqrt(torch.sum(self.mVector, dim=0)).to(device)

        self.aMatrix = torch.randn(dimOut, rank)
        stdDev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.aMatrix = nn.Parameter(self.aMatrix * stdDev).to(device)
        self.bMatrix = torch.zeros(rank, dimIn) #replace with d and k
        self.bMatrix = nn.Parameter(self.bMatrix).to(device)
    def forward(self, x):
        x = x.to(self.aMatrix.device)
        #print("FORWARD LORA DORA")
        loraMatrix = torch.matmul(self.aMatrix, self.bMatrix) + self.weight
        columnNorm = torch.sqrt(torch.sum(loraMatrix ** 2, dim=0))
        return F.linear(x, loraMatrix / columnNorm * self.mVector, self.bias)


'''
Returns pearson coefficient loss

Pearon coefficient is defined as E[(X-E[X])(Y-E[Y])]/sqrt(Var(X)Var(Y))
'''
def pearson_coefficient_loss(output, target):
    vx = output - torch.mean(output)
    vy = target - torch.mean(target)
    return -1 * torch.dot(vx, vy) * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

'''
Prints presets specified by args
'''
def print_presets(args):
    print("Settings:")
    print("Task: " + args.task)
    print("Fine tune mode: " + args.fine_tune_mode)
    print("Num of epochs: " + str(args.epochs))
    print("Num of epochs for extra loss functions: " + str(args.epochs_ft))
    print("Balance sampling factor: " + str(args.balance_sampling))
    print("Ensembling: " + args.ensembling)
    print("Lora: " + args.lora)
    print("Lora size: " + str(args.lora_size))
    print("Cosine similarity loss fine tuning: " + args.cos_sim_loss)
    print("Negative rankings loss fine tuning: " + args.neg_rankings_loss)
    print("SST hidden layers: " + str(args.sst_layers))
    print("SST hidden size: " + str(args.sst_hidden_size))
    print("PARA hidden layers: " + str(args.para_layers))
    print("PARA hidden size: " + str(args.para_hidden_size))
    print("Concat embeddings for PARA: " + str(args.para_concat))
    print("Pearson loss for STS: " + str(args.pearson_loss))
    print("STS hidden layers: " + str(args.sts_layers))
    print("STS hidden size: " + str(args.sts_hidden_size))
    print("Concat embeddings for STS: " + str(args.sts_concat))
    

'''
If args.balance_sampling != 1: Undersample from para by a factor of args.balance_sampling.
'''
def balance_sampling(sst_train_data, para_train_data, sts_train_data, args):

    n = int(len(para_train_data)/args.balance_sampling)
    para_indices = torch.randperm(len(para_train_data))[:n]
    para_train_data = [para_train_data[i] for i in para_indices]

    return sst_train_data, para_train_data, sts_train_data


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # Filtering out elements in train_data
    if args.balance_sampling != 1:
        sst_train_data, para_train_data, sts_train_data = balance_sampling(sst_train_data, para_train_data, sts_train_data, args)

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, 
                                    collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=True, batch_size=args.batch_size, 
                                    collate_fn=para_train_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)   
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # Create dataset for cosine similarity loss
    cos_sim_dataloader = None
    if args.cos_sim_loss == 'y':
        cos_sim_data = load_cos_sim_data(args.cos_sim_train) 
        cos_sim_data = SentencePairDataset(cos_sim_data, args)
        cos_sim_dataloader = DataLoader(cos_sim_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=cos_sim_data.collate_fn)

    # Create dataset for negative rankings loss
    neg_rankings_dataloader = None
    if args.neg_rankings_loss: 
        neg_rankings_data = load_neg_rankings_data(args.neg_rankings_train)
        neg_rankings_data = SentencePairDataset(neg_rankings_data, args)
        neg_rankings_dataloader = DataLoader(neg_rankings_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=neg_rankings_data.collate_fn)   

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': BERT_HIDDEN_SIZE,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)


    # Set layers for each task
    config.num_sst_layers, config.num_para_layers, config.num_sts_layers = \
        args.sst_layers, args.para_layers, args.sts_layers
    
    config.sst_hidden_size, config.para_hidden_size, config.sts_hidden_size = args.sst_hidden_size, \
        args.para_hidden_size, args.sts_hidden_size
    
    config.lora = args.lora
    config.lora_size = args.lora_size

    config.para_concat, config.sts_concat = args.para_concat, args.sts_concat

    model = MultitaskBERT(config)
    # Change model to ensembling if flag is on
    if args.ensembling == "y":
        model = BoostedBERT(config)

    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Set loss function for sts
    sts_loss_fn = lambda sts_logits, sts_labels: F.mse_loss(sts_logits, sts_labels, reduction='sum') / args.batch_size
    if args.pearson_loss == 'y':
        sts_loss_fn = pearson_coefficient_loss


    # Run extra fine tuning loss functions for bert embeddings:
    if args.cos_sim_loss == 'y' or args.neg_rankings_loss == 'y':
        print("Pretraining on additional loss functions.")
    for epoch in range(args.epochs_ft):
        model.train()
        if args.fine_tune_mode == 'full-model' and args.cos_sim_loss == 'y':
            # Train cos sim loss
            # Add dynamic lr
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(cos_sim_dataloader), epochs=args.epochs_ft) \
                if args.vary_lr == 'y' else None
            for cos_sim_batch in tqdm(cos_sim_dataloader, desc=f"PREpoch {epoch+1}/{args.epochs_ft}, Task = cosine similarity loss"):
                sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2, sts_labels = (cos_sim_batch['token_ids_1'], cos_sim_batch['attention_mask_1'],
                                                                            cos_sim_batch['token_ids_2'], cos_sim_batch['attention_mask_2'],
                                                                            cos_sim_batch['labels'])
                
                sts_ids_1 = sts_ids_1.to(device)
                sts_mask_1 = sts_mask_1.to(device)
                sts_ids_2 = sts_ids_2.to(device)
                sts_mask_2 = sts_mask_2.to(device)
                sts_labels = sts_labels.to(device).float().view(-1)

                optimizer.zero_grad()

                cos_loss = model.cos_sim_loss(sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2, sts_labels) / args.batch_size
                cos_loss.backward()
                optimizer.step()
                if args.vary_lr == 'y': 
                    scheduler.step()

        if args.fine_tune_mode == 'full-model' and args.neg_rankings_loss == 'y':
            # Train neg rankings loss
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(neg_rankings_dataloader), epochs=args.epochs_ft) \
                if args.vary_lr == 'y' else None
            for neg_rankings_batch in tqdm(neg_rankings_dataloader, desc=f"PREpoch {epoch+1}/{args.epochs_ft}, Task = negative rankings loss"):
                sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2, sts_labels = (neg_rankings_batch['token_ids_1'], neg_rankings_batch['attention_mask_1'],
                                                                            neg_rankings_batch['token_ids_2'], neg_rankings_batch['attention_mask_2'],
                                                                            neg_rankings_batch['labels'])
                
                sts_ids_1 = sts_ids_1.to(device)
                sts_mask_1 = sts_mask_1.to(device)
                sts_ids_2 = sts_ids_2.to(device)
                sts_mask_2 = sts_mask_2.to(device)
                sts_labels = sts_labels.to(device).float().view(-1)

                optimizer.zero_grad()

                neg_rankings_loss = model.multiple_negatives_ranking_loss(sts_ids_1, sts_ids_2, sts_mask_1, sts_mask_2) / args.batch_size
                neg_rankings_loss.backward()
                optimizer.step()

                if args.vary_lr == 'y': 
                    scheduler.step()

    if args.cos_sim_loss == 'y' or args.neg_rankings_loss == 'y':
        print("Pretraining over.")

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        # Train sentiment
        if args.task == "all" or args.task == "sst":
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(sst_train_dataloader), epochs=args.epochs) \
                if args.vary_lr == 'y' else None
            for sst_batch in tqdm(sst_train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}, Task = sentiment"):
                sst_ids, sst_mask, sst_labels = (sst_batch['token_ids'],
                                        sst_batch['attention_mask'], sst_batch['labels'])

                sst_ids = sst_ids.to(device)
                sst_mask = sst_mask.to(device)
                sst_labels = sst_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(sst_ids, sst_mask)

                sst_loss = F.cross_entropy(logits, sst_labels.view(-1), reduction='sum') / args.batch_size
                
                sst_loss.backward()
                optimizer.step()

                if args.vary_lr == 'y': 
                    scheduler.step()

                train_loss += sst_loss.item()
                num_batches += 1

        if args.task == "all" or args.task == "para":
            # Train paraphrase
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(para_train_dataloader), epochs=args.epochs) \
                if args.vary_lr == 'y' else None
            for para_batch in tqdm(para_train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}, Task = paraphrase"):
                para_ids_1, para_mask_1, para_ids_2, para_mask_2, para_labels = (para_batch['token_ids_1'], para_batch['attention_mask_1'],
                                                                                para_batch['token_ids_2'], para_batch['attention_mask_2'],
                                                                                para_batch['labels'])
                para_ids_1 = para_ids_1.to(device)
                para_mask_1 = para_mask_1.to(device)
                para_ids_2 = para_ids_2.to(device)
                para_mask_2 = para_mask_2.to(device)
                para_labels = para_labels.to(device).float().view(-1)

                optimizer.zero_grad()
                para_logits = model.predict_paraphrase(para_ids_1, para_mask_1, para_ids_2, para_mask_2)
                para_logits = torch.squeeze(para_logits, 1)
                para_loss = F.binary_cross_entropy_with_logits(para_logits, para_labels, reduction='sum') / args.batch_size
                
                para_loss.backward()
                optimizer.step()

                if args.vary_lr == 'y': 
                    scheduler.step()

                train_loss += para_loss.item()
                num_batches += 1

        if args.task == "all" or args.task == "sts":
            # Train similarity
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(sts_train_dataloader), epochs=args.epochs) \
                if args.vary_lr == 'y' else None
            for sts_batch in tqdm(sts_train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}, Task = similarity"):
                sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2, sts_labels = (sts_batch['token_ids_1'], sts_batch['attention_mask_1'],
                                                                            sts_batch['token_ids_2'], sts_batch['attention_mask_2'],
                                                                            sts_batch['labels'])
                sts_ids_1 = sts_ids_1.to(device)
                sts_mask_1 = sts_mask_1.to(device)
                sts_ids_2 = sts_ids_2.to(device)
                sts_mask_2 = sts_mask_2.to(device)
                sts_labels = sts_labels.to(device).float()

                optimizer.zero_grad()

                sts_logits = model.predict_similarity(sts_ids_1, sts_mask_1, sts_ids_2, sts_mask_2)
                sts_logits = torch.squeeze(sts_logits, 1)
                sts_loss = sts_loss_fn(sts_logits, sts_labels)

                sts_loss.backward()
                train_loss += sts_loss.item()
                num_batches += 1

                if args.contrastive_learning == 'y':
                    emb_1 = model.bert(sts_ids_1, sts_mask_1)['pooler_output']
                    emb_2 = model.bert(sts_ids_2, sts_mask_2)['pooler_output']
                    ntxent_loss = model.compute_ntxent_loss(emb_1, emb_2)

                    ntxent_loss.backward()
                    optimizer.step()

                    train_loss += ntxent_loss.item()
                    num_batches += 1
                
                optimizer.step()
                if args.vary_lr == 'y': 
                    scheduler.step()
        
        train_loss = train_loss / (num_batches)
        overall_dev_acc = 0
        if args.task == "sst":
            sentiment_accuracy, _, _, _, _, _ = model_eval_sst(sst_dev_dataloader, model, device)
            overall_dev_acc = sentiment_accuracy
        elif args.task == "para":
            paraphrase_accuracy, _, _ = model_eval_para(para_dev_dataloader, model, device)
            overall_dev_acc = paraphrase_accuracy
        elif args.task == "sts":
            sts_corr, _, _ = model_eval_sts(sts_dev_dataloader, model, device)
            overall_dev_acc = sts_corr
        else:
            sentiment_accuracy, sst_y_pred, sst_sent_ids, paraphrase_accuracy, para_y_pred, para_sent_ids, sts_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
            overall_dev_acc = (sentiment_accuracy + paraphrase_accuracy + sts_corr) / 3

        if overall_dev_acc > best_dev_acc:
            best_dev_acc = overall_dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch+1}: train loss :: {train_loss :.3f}, dev acc :: {overall_dev_acc :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        if args.ensembling == 'y':
            model = BoostedBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        if args.task == "all":
            dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
                dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
                dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                        para_dev_dataloader,
                                                                        sts_dev_dataloader, model, device)

            test_sst_y_pred, \
                test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                    model_eval_test_multitask(sst_test_dataloader,
                                            para_test_dataloader,
                                            sts_test_dataloader, model, device)

            with open(args.sst_dev_out, "w+") as f:
                print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.sst_test_out, "w+") as f:
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.para_dev_out, "w+") as f:
                print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.para_test_out, "w+") as f:
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(test_para_sent_ids, test_para_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.sts_dev_out, "w+") as f:
                print(f"dev sts corr :: {dev_sts_corr :.3f}")
                f.write(f"id \t Predicted_Similiarity \n")
                for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.sts_test_out, "w+") as f:
                f.write(f"id \t Predicted_Similiarity \n")
                for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                    f.write(f"{p} , {s} \n")
        elif args.task == "sst":
            dev_sentiment_accuracy, _, dev_sst_y_pred, _, _, dev_sst_sent_ids = model_eval_sst(sst_dev_dataloader,
                                                                        model, device)
            sst_y_pred, sst_sent_ids = model_eval_test_sst(sst_test_dataloader, model, device)

            with open(args.sst_dev_out, "w+") as f:
                print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                    f.write(f"{p} , {s} \n")
            with open(args.sst_test_out, "w+") as f:
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(sst_y_pred, sst_sent_ids):
                    f.write(f"{p} , {s} \n")
        elif args.task == "para":
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids \
                = model_eval_para(para_dev_dataloader, model, device)

            test_para_y_pred, test_para_sent_ids = \
                    model_eval_test_para(para_test_dataloader, model, device)

            with open(args.para_dev_out, "w+") as f:
                print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.para_test_out, "w+") as f:
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(test_para_sent_ids, test_para_y_pred):
                    f.write(f"{p} , {s} \n")
        else:
            dev_sts_corr, _, dev_sts_y_pred, _, _, dev_sts_sent_ids = model_eval_sts(sts_dev_dataloader, model, device)
            #dev_sts_corr, _, dev_sts_y_pred, _, _, dev_sts_sent_ids = model_eval_sts(sts_dev_dataloader, model, device)

            test_sts_y_pred, test_sts_sent_ids = \
                    model_eval_test_sts(sts_test_dataloader, model, device)

            with open(args.sts_dev_out, "w+") as f:
                print(f"dev sts corr :: {dev_sts_corr :.3f}")
                f.write(f"id \t Predicted_Similarity \n")
                for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                    f.write(f"{p} , {s} \n")

            with open(args.sts_test_out, "w+") as f:
                f.write(f"id \t Predicted_Similarity \n")
                for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                    f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    # Select task: "all" does all tasks
    parser.add_argument("--task", type=str, default = "all")
    # FLAGS for testing different models

    # 1a. Set num linear layers for sst
    parser.add_argument("--sst_layers", type=int,
                        help='num linear layers for sst',
                        default = 2)
    
    # 1b. Set num linear layers for para
    parser.add_argument("--para_layers", type=int,
                        help='num linear layers for para',
                        default= 2)
    
    # 1c. Set num linear layers for sts
    parser.add_argument("--sts_layers", type = int,
                        help='num linear layers for sts',
                        default = 0)
    

    # 2. Set cosine similarity loss for similarity task
    parser.add_argument("--cos_sim_loss", type=str,
                        choices=('y', 'n'),
                        help='cosine similarity loss for embeddings',
                        default = 'y')
    # 3. Set neg ranking loss for similarity task
    parser.add_argument("--neg_rankings_loss", type=str,
                        help='neg ranking loss for embeddings',
                        choices=('y', 'n'),
                        default = 'y')
    
    #4. Num of epochs for cosine similarity loss and neg ranking loss
    parser.add_argument("--epochs_ft", type=int,
                        help = 'num of epochs for cosine similarity loss and neg ranking loss',
                        default = 1)

    # 5. Balance sampling
    parser.add_argument("--balance_sampling", type=int,
                        help='choose what factor by which to reduce number of PARA examples',
                        default = 1)
    
    # 6. Boosted bert
    parser.add_argument("--ensembling", type=str,
                        choices=('y', 'n'),
                        default = 'n')
    
    # 7. Hidden size for linear layers
    parser.add_argument("--sst_hidden_size", type=int,
                        default = 100)
    
    parser.add_argument("--para_hidden_size", type=int,
                        default = 100)
    
    parser.add_argument("--sts_hidden_size", type=int,
                        default = 5)
    
    # 8. Lora model
    parser.add_argument("--lora", type=str,
                        choices=('y', 'n'),
                        default = 'n')
    
    parser.add_argument("--lora_size", type=int,
                        default = 100)

    # 9. Concatenate embeddings for para and sts
    parser.add_argument("--para_concat",
                        type=str,
                        help='Concatenate bert embeddings for para',
                        choices=('y','n'),
                        default='y')
    parser.add_argument("--sts_concat",
                        type=str,
                        help='Concatenate bert embeddings for sts',
                        choices=('y','n'),
                        default='n')
    
    # 10. Pearson loss for sts
    parser.add_argument("--pearson_loss",
                        type=str,
                        help="Use Pearson coefficient loss for STS task",
                        choices=('y','n'),
                        default='y')
    
    # 11. Skip training and run test_multitask
    parser.add_argument("--eval",
                        type=str,
                        help="Skip training and run test_multitask",
                        choices=('y','n'),
                        default='n')
    
    # 12. Variable learning rate
    parser.add_argument("--vary_lr",
                         type = str,
                         choices=('y','n'),
                         default='n')
    
    # 12. contrastive_learning
    parser.add_argument("--contrastive_learning",
                         type = str,
                         choices=('y','n'),
                         default='y')
    parser.add_argument("--contrastive_weight",
                         type = float,
                         default=0.5)

    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--cos-sim-train", type=str, default="data/cos-sim-train.csv")
    parser.add_argument("--neg-rankings-train", type=str, default="data/neg-rankings-train.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    if args.eval == 'y':
        test_multitask(args)
    else:
        seed_everything(args.seed)  # Fix the seed for reproducibility.
        print_presets(args)
        train_multitask(args)
        test_multitask(args)
        print_presets(args)