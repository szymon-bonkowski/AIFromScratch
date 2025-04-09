import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 5000
embedding_dim = 128
num_heads = 4
num_layers = 2
seq_length = 128

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim musi być podzielny przez liczbę głowic."
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.size()

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, values)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embedding_dim)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        ff_out = self.ff(x)
        x = self.norm2(ff_out + x)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, seq_length):
        super(MiniGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_length, embedding_dim)
        
        self.layers = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.fc_out(x)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, embedding_dim, num_heads, num_layers, seq_length).to(device)
    
    sample_input = torch.randint(0, vocab_size, (2, seq_length)).to(device)
    logits = model(sample_input)
    print("Output logits shape:", logits.shape)
