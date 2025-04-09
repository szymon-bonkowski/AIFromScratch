import json
import re
import torch
from torch.utils.data import Dataset

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

class TextDataset(Dataset):
    def __init__(self, file_path, vocab_path, seq_length=128):
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_idx = json.load(f)
        
        self.seq_length = seq_length
        
        self.tokens = tokenize(self.text)
        self.token_indices = [self.token_to_idx.get(token, self.token_to_idx["<UNK>"]) for token in self.tokens]
        
    def __len__(self):
        return len(self.token_indices) // self.seq_length
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        input_seq = torch.tensor(self.token_indices[start:end], dtype=torch.long)
        target_seq = torch.tensor(self.token_indices[start+1:end+1], dtype=torch.long)
        return input_seq, target_seq

if __name__ == "__main__":
    dataset = TextDataset("train.txt", "vocab.json", seq_length=128)
    print("Number of sequences in dataset:", len(dataset))
    sample_input, sample_target = dataset[0]
    print("Przykładowa sekwencja wejściowa:", sample_input)
    print("Przykładowa sekwencja docelowa:", sample_target)
