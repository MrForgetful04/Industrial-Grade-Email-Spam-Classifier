import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os


data_path = "data/enron_spam_balanced.csv"  # balanced subset (~15k)
df = pd.read_csv(data_path)
print(f"Loaded dataset with {len(df)} emails")


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()
model.to(device)


def compute_embeddings(texts, tokenizer, model, device='cpu', batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=300  # truncate to speed up
            )
            for k in encoded:
                encoded[k] = encoded[k].to(device)
            output = model(**encoded)
            batch_emb = output.last_hidden_state[:,0,:].cpu().numpy()  
            embeddings.append(batch_emb)
    return np.vstack(embeddings)

print("Computing DistilBERT embeddings...")
X_bert = compute_embeddings(df['text'].tolist(), tokenizer, model, device=device, batch_size=32)


os.makedirs("data", exist_ok=True)
emb_path = "data/enron_bert_embeddings_balanced.npy"
np.save(emb_path, X_bert)
print(f"Saved embeddings to {emb_path}")
