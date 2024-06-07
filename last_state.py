import torch

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
