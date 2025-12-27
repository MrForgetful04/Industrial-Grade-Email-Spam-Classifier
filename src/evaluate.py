import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from joblib import load
from scipy.sparse import hstack
import re

# ----------------------------
# Load artifacts
# ----------------------------
tfidf = load("models/tfidf_vectorizer.joblib")
scaler = load("models/handcrafted_scaler.joblib")
model_lr = load("models/logreg_model.joblib")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
bert_model.to(device)

# ----------------------------
# Helper: handcrafted features
# ----------------------------
def extract_features(email_text):
    num_urls = len(re.findall(r'http[s]?://', email_text))
    has_html = int(bool(re.search(r'<.*?>', email_text)))
    subject_len = len(email_text.split("\n")[0])  # assume first line = subject
    body_len = len(email_text)
    num_exclamations = email_text.count('!')
    num_caps = sum(1 for c in email_text if c.isupper())
    return np.array([[num_urls, has_html, subject_len, body_len, num_exclamations, num_caps]])

# ----------------------------
# Compute BERT embedding
# ----------------------------
def compute_bert(text):
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=300)
    for k in encoded:
        encoded[k] = encoded[k].to(device)
    with torch.no_grad():
        output = bert_model(**encoded)
    return output.last_hidden_state[:,0,:].cpu().numpy()  # CLS token

# ----------------------------
# Predict function
# ----------------------------
def predict_email(text):
    X_tfidf = tfidf.transform([text])
    X_hand = scaler.transform(extract_features(text))
    X_bert = compute_bert(text)
    X_final = hstack([X_tfidf, X_bert, X_hand])
    y_pred = model_lr.predict(X_final)[0]
    y_prob = model_lr.predict_proba(X_final)[0][1]
    return y_pred, y_prob

# ----------------------------
# Example usage
# ----------------------------
email_text = """Hi Team,

Please find attached the agenda for our Friday meeting at 10 AM.  
Let me know if you have any questions or topics to add.

Best regards,  
Alice
"""

label, prob = predict_email(email_text)
print("Predicted label:", "Spam" if label else "Ham")
print("Probability of spam:", prob)
