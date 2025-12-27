import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from joblib import dump


data_path = "data/enron_spam_balanced.csv" 
df = pd.read_csv(data_path)
print(f"Loaded dataset with {len(df)} emails")

handcrafted_features = ['num_urls', 'has_html', 'subject_len', 'body_len', 'num_exclamations', 'num_caps']
X_handcrafted = df[handcrafted_features].values
y = df['label'].values


tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=200000)
X_tfidf = tfidf.fit_transform(df['text'])


X_bert = np.load("data/enron_bert_embeddings_balanced.npy")
print(f"Loaded precomputed BERT embeddings with shape: {X_bert.shape}")


scaler = StandardScaler()
X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)
X_final = hstack([X_tfidf, X_bert, X_handcrafted_scaled])


X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)


# Evaluating the balanced dataset

y_pred = model_lr.predict(X_test)
y_prob = model_lr.predict_proba(X_test)[:,1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))


output_folder = "models"
os.makedirs(output_folder, exist_ok=True)

dump(model_lr, os.path.join(output_folder, "logreg_model.joblib"))
dump(tfidf, os.path.join(output_folder, "tfidf_vectorizer.joblib"))
dump(scaler, os.path.join(output_folder, "handcrafted_scaler.joblib"))
print(f"\nSaved TF-IDF, LR model, and scaler to {output_folder}")
