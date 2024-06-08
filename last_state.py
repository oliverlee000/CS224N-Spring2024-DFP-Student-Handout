import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import BoostedBERT, MultitaskBERT
from optimizer import AdamW
from datasets import (
    SentenceClassificationTestDataset,
    SentencePairTestDataset,
    load_multitask_data
)
from evaluation import model_eval_multitask, model_eval_sst, model_eval_para, model_eval_sts

# Define a function to load the saved model state
def load_model(filepath):
    saved = torch.load(filepath)
    return saved

# Load the saved state
saved_state = load_model('full-model-10-1e-05-multitask.pt')

# Extract the saved information
model_state = saved_state['model']
optimizer_state = saved_state['optim']
args = saved_state['args']
config = saved_state['model_config']
system_rng = saved_state['system_rng']
numpy_rng = saved_state['numpy_rng']
torch_rng = saved_state['torch_rng']

# Print the saved arguments and configurations
print("Training Arguments:", args)
print("Model Configuration:", config)

# Load the random states to ensure reproducibility
random.setstate(system_rng)
np.random.set_state(numpy_rng)
torch.random.set_rng_state(torch_rng)

# Recreate the model and optimizer with the saved states
device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

# Initialize your model and optimizer (ensure that MultitaskBERT and BoostedBERT are properly imported and defined)
if args.ensembling == 'y':
    model = BoostedBERT(config)
else:
    model = MultitaskBERT(config)

model.load_state_dict(model_state)
model.to(device)

optimizer = AdamW(model.parameters(), lr=args.lr)
optimizer.load_state_dict(optimizer_state)

print("Model and optimizer states loaded successfully.")

# Evaluate the model
def evaluate_model(model, args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    model = model.to(device)

    # Load your datasets and dataloaders here
    sst_test_data, num_labels, para_test_data, sts_test_data = \
        load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

    sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
    para_test_data = SentencePairTestDataset(para_test_data, args)
    sts_test_data = SentencePairTestDataset(sts_test_data, args)

    sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sst_test_data.collate_fn)
    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)
    sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_test_data.collate_fn)

    if args.task == "all":
        sentiment_accuracy, sst_y_pred, sst_sent_ids, paraphrase_accuracy, para_y_pred, para_sent_ids, sts_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(
            sst_test_dataloader, para_test_dataloader, sts_test_dataloader, model, device)
        print(f"Test Sentiment Accuracy: {sentiment_accuracy}")
        print(f"Test Paraphrase Accuracy: {paraphrase_accuracy}")
        print(f"Test STS Correlation: {sts_corr}")
    elif args.task == "sst":
        sentiment_accuracy, _, sst_y_pred, _, _, sst_sent_ids = model_eval_sst(sst_test_dataloader, model, device)
        print(f"Test Sentiment Accuracy: {sentiment_accuracy}")
    elif args.task == "para":
        paraphrase_accuracy, para_y_pred, para_sent_ids = model_eval_para(para_test_dataloader, model, device)
        print(f"Test Paraphrase Accuracy: {paraphrase_accuracy}")
    else:
        sts_corr, _, sts_y_pred, _, _, sts_sent_ids = model_eval_sts(sts_test_dataloader, model, device)
        print(f"Test STS Correlation: {sts_corr}")

# Evaluate the model
evaluate_model(model, args)
