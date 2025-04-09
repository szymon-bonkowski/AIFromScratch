import json
import re
from collections import Counter

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def build_vocab(token_list, min_freq=1):
    token_counts = Counter(token_list)
    vocab_tokens = [token for token, count in token_counts.items() if count >= min_freq]
    vocab_tokens = sorted(vocab_tokens)
    vocab_tokens = ["<PAD>", "<UNK>"] + vocab_tokens
    
    token_to_idx = {token: idx for idx, token in enumerate(vocab_tokens)}
    return token_to_idx

def save_vocab(token_to_idx, path="vocab.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(token_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {path}")

def main():
    with open("train.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenize(text)
    print(f"Liczba tokenów: {len(tokens)}")
    
    token_to_idx = build_vocab(tokens, min_freq=1)
    print(f"Rozmiar słownika: {len(token_to_idx)}")
    
    save_vocab(token_to_idx)

if __name__ == "__main__":
    main()
