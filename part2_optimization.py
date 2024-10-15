import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt 
import random 
import numpy as np  
import optuna
import logging 
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings 
from tqdm.notebook import tqdm
warnings.filterwarnings("ignore") 


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.replace("-", float("nan"))
        self.data = self.data.dropna() 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = torch.tensor(self.data.iloc[idx, 0], dtype=torch.float32)
        x = float(self.data.iloc[idx, 1])
        y = float(self.data.iloc[idx, 2])
        xy = torch.tensor([x, y], dtype=torch.float32)
        return t, xy 


def loss_fn(outputs, labels):
    return torch.mean((outputs - labels)**2)

def predict_vals(model, dataframe):
    dataframe = pd.read_csv(dataframe)
    model.eval()
    t_vals = dataframe.iloc[:, 0] 
    t_vals = torch.tensor(t_vals, dtype=torch.float32) 
    with torch.no_grad():
        outputs = model(t_vals)
    predictions = outputs.detach().numpy() 
    dataframe = dataframe.replace("-", float("nan"))
    for i in range(len(dataframe)):
        if pd.isna(dataframe.iloc[i, 1]):
            dataframe.iloc[i, 1] = predictions[i, 0]
        if pd.isna(dataframe.iloc[i, 2]):
            dataframe.iloc[i, 2] = predictions[i, 1]
    dataframe['x'] = dataframe['x'].astype(float)
    dataframe['y'] = dataframe['y'].astype(float)
    return dataframe


optimizers = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop
}


def objective(trial):
    # Define hyperparameters to optimize
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"])
    
    # Learning rate (log-uniform distribution)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    
    # Number of epochs
    n_epochs = trial.suggest_int("n_epochs", 50, 300)
    
    # Batch size
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    
    # Model architecture
    n_layers = trial.suggest_int("n_layers", 2, 5)
    layer_sizes = [trial.suggest_int(f"layer_{i}_size", 32, 256) for i in range(n_layers)]
    
    # Dropout rate
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
    
    # Activation function
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "ELU"])
    
    # Weight decay for regularization
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-3)
    
    # Create model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(1, layer_sizes[0]))
            for i in range(1, len(layer_sizes)):
                self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.layers.append(nn.Linear(layer_sizes[-1], 2))
            self.dropout = nn.Dropout(dropout_rate)
            self.activation = getattr(nn, activation)()

        def forward(self, x):
            x = x.view(-1, 1)
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
                x = self.dropout(x)
            x = self.layers[-1](x)
            return x

    net = Net()
    optimizer = optimizers[optimizer_name](net.parameters(), lr=lr)
    
    # Create data loader
    dataset = TimeSeriesDataset("data.csv")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Report intermediate objective value
        trial.report(running_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return running_loss


sampler = optuna.samplers.TPESampler(seed=42) 
study = optuna.create_study(direction="minimize",study_name="Optimization",sampler=sampler)
study.optimize(objective, n_trials=100,show_progress_bar=True)  # Adjust n_trials as needed 

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))



