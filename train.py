import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json

from mini_gpt import MiniGPT

from dataset import TextDataset

vocab_path = "vocab.json"
data_path = "train.txt"
seq_length = 128
batch_size = 16
num_epochs = 5
learning_rate = 3e-4
save_every = 100

with open(vocab_path, "r", encoding="utf-8") as f:
    token_to_idx = json.load(f)
vocab_size = len(token_to_idx)

embedding_dim = 128
num_heads = 4
num_layers = 2

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    model = MiniGPT(vocab_size, embedding_dim, num_heads, num_layers, seq_length).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    dataset = TextDataset(data_path, vocab_path, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            logits = model(batch_inputs)
            loss = criterion(logits.view(-1, vocab_size), batch_targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if global_step % save_every == 0:
                print(f"Step: {global_step}, Loss: {loss.item():.4f}")
                checkpoint_path = f"checkpoint_step_{global_step}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")
            global_step += 1
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")
    
    final_model_path = "mini_gpt_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    train()
