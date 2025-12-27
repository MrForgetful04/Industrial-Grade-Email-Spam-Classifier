import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from joblib import load
from scipy.sparse import hstack
import re
import random


tfidf = load("models/tfidf_vectorizer.joblib")
scaler = load("models/handcrafted_scaler.joblib")
model_lr = load("models/logreg_model.joblib")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
bert_model.eval()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
bert_model.to(device)

def extract_features(email_text):
    num_urls = len(re.findall(r'http[s]?://', email_text))
    has_html = int(bool(re.search(r'<.*?>', email_text)))
    subject_len = len(email_text.split("\n")[0])  # assume first line = subject
    body_len = len(email_text)
    num_exclamations = email_text.count('!')
    num_caps = sum(1 for c in email_text if c.isupper())
    return np.array([[num_urls, has_html, subject_len, body_len, num_exclamations, num_caps]])


def compute_bert(text):
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=300)
    for k in encoded:
        encoded[k] = encoded[k].to(device)
    with torch.no_grad():
        output = bert_model(**encoded)
    return output.last_hidden_state[:,0,:].cpu().numpy()  


def predict_email(text):
    X_tfidf = tfidf.transform([text])
    X_hand = scaler.transform(extract_features(text))
    X_bert = compute_bert(text)
    X_final = hstack([X_tfidf, X_bert, X_hand])
    y_pred = model_lr.predict(X_final)[0]
    y_prob = model_lr.predict_proba(X_final)[0][1]
    return y_pred, y_prob


# Example usage

ham_emails = [
    "hi team please find attached the agenda for tomorrow’s meeting let me know if you have any questions or additional topics",
    "just confirming that we are still on for lunch with the client tomorrow at 1 pm let me know if anything changes",
    "re contracts and credit thanks i ll include it in the master file original message from farmer daren j sent thursday january 10 2002 8 02 am",
    "we need to finalize the quarterly report before friday please send me your sections asap",
    "hey, are we still on for the project call this afternoon? let me know"
]


spam_emails = [
    "claim your free bitcoin today by signing up with this exclusive link and start earning instantly click here to claim your bonus now",
    "buy authentic luxury watches at 90% off for the next 24 hours only visit our website and secure your deal before it expires",
    "your prescription is ready low cost prescription medications shipped overnight click here to order",
    "get that new car 8434 people nowthe weather or climate in any particular environment can change and affect what people eat",
    "introducing doctor formulated hgh human growth hormone increase energy and muscle strength click here to learn more"
]

email_examples = ham_emails + spam_emails

def run_prediction(index=None):
    """
    Predict a single email.
    If index is provided, use that email from email_examples.
    Otherwise, select a random email.
    """
    if index is not None and 0 <= index < len(email_examples):
        email_text = email_examples[index]
    else:
        email_text = random.choice(email_examples)
    
    label, prob = predict_email(email_text)
    
    print("Email text:\n", email_text, "\n")
    print("Predicted label:", "Spam" if label else "Ham")
    print("Probability of spam:", prob)

run_prediction()  # Random email
run_prediction(index=2)