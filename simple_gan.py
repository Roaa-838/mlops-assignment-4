import os
# Fix for the Windows/Anaconda DLL kernel crash
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as  pd
import numpy as np
from sklearn.datasets import load_digits
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch


# 1. Tell the script to talk directly to the live dashboard
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Set the experiment name
mlflow.set_experiment("Assignment3_GAN_Tracking")
# ==========================================
# 1. GENERATE THE DATASET
# ==========================================
print("Fetching the 8x8 digits dataset...")
digits = load_digits()
df_initial = pd.DataFrame(digits.data)
df_initial.to_csv("simple_digits.csv", index=False)
print("Success! simple_digits.csv saved.")

# ==========================================
# 2. LOAD AND PREPARE DATA FOR GAN
# ==========================================
print("Loading data into the GAN...")
df = pd.read_csv("simple_digits.csv")
real_data = torch.tensor(df.values, dtype=torch.float32) / 16.0

# ==========================================
# 3. DEFINE THE GAN ARCHITECTURE
# ==========================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
# ==========================================
# 4. SET HYPERPARAMETERS 

lr = 0.00001
batch_size = 32
num_epochs = 800
latent_dim = 32

# ==========================================
# 5. INITIALIZE MODELS & OPTIMIZERS
# ==========================================
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()

# Now the optimizers will actually listen to your 'lr' variable!
d_optimizer = optim.Adam(D.parameters(), lr=lr)
g_optimizer = optim.Adam(G.parameters(), lr=lr)

# ==========================================
# 6. TRAINING LOOP (WITH MLFLOW)
# ==========================================
with mlflow.start_run():
    # 1. Log your ID tag
    mlflow.set_tag("student_id", "202202079") # Make sure to put your real ID!
    
    # 2. Log the parameters
    mlflow.log_params({
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size
    })

    print("Starting training...")
    for epoch in range(num_epochs):
        
        # Train Discriminator
        indices = torch.randint(0, len(real_data), (batch_size,))
        real_batch = real_data[indices]
        
        noise = torch.randn(batch_size, latent_dim)
        fake_batch = G(noise)
        
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        d_loss_real = criterion(D(real_batch), real_labels)
        d_loss_fake = criterion(D(fake_batch.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        noise = torch.randn(batch_size, latent_dim)
        fake_batch = G(noise)
        g_loss = criterion(D(fake_batch), real_labels)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        # 3. Live Logging: At the end of the epoch, log the metrics!
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            mlflow.log_metrics({
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item()
            }, step=epoch)
            
    # 4. Save the Model Artifacts at the very end
    print("Training complete. Saving model to MLflow...")
    mlflow.pytorch.log_model(G, "generator_model")