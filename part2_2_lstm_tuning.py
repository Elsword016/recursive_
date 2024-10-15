import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.replace("-", float("nan"))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        t = torch.tensor(self.data.iloc[idx, 0], dtype=torch.float32)
        x = self.data.iloc[idx, 1]
        y = self.data.iloc[idx, 2]
        
        # Normalize t to [0, 1] range (assuming max t is 20)
        t = t / 20
        
        # Create mask for original missing values
        mask = torch.tensor([not pd.isna(self.data.iloc[idx, 1]), not pd.isna(self.data.iloc[idx, 2])], dtype=torch.float32)
        
        # Convert x and y to float
        x = float(x)
        y = float(y)
        
        xy = torch.tensor([x, y], dtype=torch.float32)
        return t, xy, mask


class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=2, num_layers=3, dropout=0.2):
        super(TimeSeriesModel, self).__init__()
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embed(x)
        x, _ = self.lstm(x)
        return self.output(x.squeeze(1))
    
    def extrapolate(self, t):
        t_normalized = t / 100
        return self.forward(t_normalized)


def masked_mse_loss(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction='none')
    loss = loss * mask
    return loss.sum() / mask.sum()


def train_model(model, train_set, val_set, epochs=300, lr=0.001, patience=20, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_loss = []
    validation_loss = []
    lr_history = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0
        for t, xy, mask in train_set:
            t, xy, mask = t.to(device), xy.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(t)
            loss = masked_mse_loss(pred, xy, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for t, xy, mask in val_set:
                t, xy, mask = t.to(device), xy.to(device), mask.to(device)
                pred = model(t)
                val_loss += masked_mse_loss(pred, xy, mask).item()
        
        train_loss /= len(train_set)
        val_loss /= len(val_set)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_extrapolate.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    return best_val_loss


def objective(trial):
    # Define hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Create model with suggested hyperparameters
    model = TimeSeriesModel(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    
    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train the model
    best_val_loss = train_model(model, train_loader, val_loader, epochs=100, lr=lr, patience=20, device=device)
    
    return best_val_loss


# Prepare your dataset
dataset = TimeSeriesDataset('data.csv')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run Optuna study
sampler = optuna.samplers.TPESampler(seed=42)  
study = optuna.create_study(direction='minimize',study_name="LSTM_Tuning")
study.optimize(objective, n_trials=20,show_progress_bar=True)


print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


import json 
with open('best_config_lstm.json', 'w') as f:
    json.dump(study.best_params, f, indent=4)

print("Best configuration saved to 'best_config_lstm.json'")


