import torch
import torch.nn.functional as F
import json
import re
from mini_gpt import MiniGPT

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def detokenize(token_list):
    return " ".join(token_list)

def load_vocab(vocab_path="vocab.json"):
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_idx = json.load(f)
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token

def encode(text, token_to_idx):
    tokens = tokenize(text)
    return [token_to_idx.get(token, token_to_idx["<UNK>"]) for token in tokens]

def decode(indices, idx_to_token):
    tokens = [idx_to_token.get(idx, "<UNK>") for idx in indices]
    return detokenize(tokens)

def generate(model, prompt, token_to_idx, idx_to_token, seq_length=128, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device
    input_indices = encode(prompt, token_to_idx)
    if len(input_indices) == 0:
        input_indices = [token_to_idx["<UNK>"]]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
    model_input = input_tensor
    for _ in range(max_new_tokens):
        if model_input.shape[1] > seq_length:
            model_input = model_input[:, -seq_length:]
        logits = model(model_input)
        last_logits = logits[0, -1, :]
        next_token = torch.argmax(last_logits).unsqueeze(0)
        model_input = torch.cat([model_input, next_token.unsqueeze(0)], dim=1)
    generated_indices = model_input[0].tolist()
    generated_text = decode(generated_indices, idx_to_token)
    return generated_text

if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = "vocab.json"
    token_to_idx, idx_to_token = load_vocab(vocab_path)
    vocab_size = len(token_to_idx)
    embedding_dim = 128
    num_heads = 4
    num_layers = 2
    seq_length = 128
    model = MiniGPT(vocab_size, embedding_dim, num_heads, num_layers, seq_length).to(device)
    checkpoint_path = "mini_gpt_final.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Model załadowany.")
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Podaj prompt: ")
    generated_text = generate(model, prompt, token_to_idx, idx_to_token, seq_length, max_new_tokens=50)
    print("\n--- Wygenerowany tekst ---\n")
    print(generated_text)
