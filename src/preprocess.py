import pandas as pd
import numpy as np
import re
import os

# ----------------------------
# 1) Load Enron Dataset
# ----------------------------
raw_path = "data/enron_spam_data.csv"  # adjust path
df = pd.read_csv(raw_path)
df.rename(columns={"Spam/Ham": "label"}, inplace=True)
df['label'] = df['label'].map({"ham": 0, "spam": 1})

# ----------------------------
# 2) Industrial Cleaning Function
# ----------------------------
def clean_email(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)        # HTML tags
    text = re.sub(r"http\S+", " <URL> ", text)  # URLs
    text = re.sub(r"\S+@\S+", " <EMAIL> ", text)  # emails
    text = re.sub(r"[^a-z0-9\s]", " ", text) # non-alphanumeric chars
    text = re.sub(r"\s+", " ", text).strip() # extra spaces
    return text

df['subject_clean'] = df['Subject'].fillna("").apply(clean_email)
df['body_clean'] = df['Message'].fillna("").apply(clean_email)

# ----------------------------
# 3) Handcrafted Feature Extraction
# ----------------------------
def extract_features(df):
    df['num_urls'] = df['Message'].apply(lambda x: len(re.findall(r"http\S+", str(x))))
    df['has_html'] = df['Message'].apply(lambda x: 1 if bool(re.search(r"<.*?>", str(x))) else 0)
    df['subject_len'] = df['subject_clean'].apply(len)
    df['body_len'] = df['body_clean'].apply(len)
    df['num_exclamations'] = df['Message'].apply(lambda x: str(x).count('!'))
    df['num_caps'] = df['Message'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
    return df

df = extract_features(df)

# ----------------------------
# 4) Combine subject + body
# ----------------------------
df['text'] = df['subject_clean'] + " " + df['body_clean']

# ----------------------------
# 5) Write cleaned dataset to CSV
# ----------------------------
# Get folder of raw CSV
folder = os.path.dirname(raw_path)
clean_path = os.path.join(folder, "enron_spam_cleaned.csv")

# Save only useful columns
columns_to_save = [
    'label', 'subject_clean', 'body_clean', 'text',
    'num_urls', 'has_html', 'subject_len', 'body_len',
    'num_exclamations', 'num_caps'
]

df[columns_to_save].to_csv(clean_path, index=False)
print(f"Cleaned dataset saved to: {clean_path}")
