# Industrial-Grade Email Spam Classifier


## Problem Statement

Email spam wastes time, resources, and exposes users to phishing or malware. Existing classifiers often assume well-structured text, which fails in real-world scenarios.

This project builds a **robust classifier for unstructured, noisy emails** — emails with inconsistent grammar, formatting, or "drunk-looking" text — demonstrating practical ML for messy, real-world data.

---

## Dataset Used

**Primary dataset:** [Enron Email Dataset (Kaggle)](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

**Balanced subset logic:**

- Original: ~3,671 ham vs. >13,000 spam emails.
- Created a **balanced subset**: randomly sampled **half of ham and half of spam**, yielding ~16,857 emails.
- Ensures model does not overfit to the majority class.

**Why this dataset:**

- Reflects **real-world noise** — irregular formatting, punctuation, and inconsistent grammar.
- Conventional clean datasets would make the task trivial; this dataset forces a model to handle ambiguity.

---

## Pipeline Overview

The pipeline leverages a **hybrid feature approach**: semantic embeddings, TF-IDF, and handcrafted email features.

**High-level workflow:**

```
Raw Enron Emails
       |
       v
Data Cleaning & Preprocessing
       |
       v
Feature Engineering:
    - TF-IDF (n-grams, min_df=3)
    - Handcrafted Features (num_urls, HTML, exclamations, caps)
    - DistilBERT Embeddings
       |
       v
Feature Combination (TF-IDF + BERT + Scaled Handcrafted)
       |
       v
Logistic Regression Classifier
       |
       v
Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)

```

**Rationale for choices:**

- **TF-IDF:** Efficient for capturing surface-level patterns.
- **Handcrafted features:** Exploit known spam cues (URLs, capitalization, punctuation).
- **DistilBERT:** Captures semantic context and meaning.
- **Logistic Regression:** Balances interpretability, scalability, and robustness. Alternatives like Random Forest or deep networks either struggle with sparse features or are computationally expensive.

---

## How to Run

### 1. Training

```bash
python src/train.py

```

- Combines TF-IDF, DistilBERT embeddings, and handcrafted features.
- Splits into train/test.
- Trains Logistic Regression.
- Outputs models and vectorizers in `models/`.

---

### 2. Precompute Embeddings

```bash
python src/precompute_embeddings.py

```

- Generates DistilBERT embeddings (`.npy`) for all emails.
- Supports GPU acceleration.
- Batch size adjustable (e.g., 16, 32).

---

### 3. Prediction / Evaluation

```python
from src.test import predict_email

email_text = "hi team please find attached the agenda for tomorrow’s meeting"
label, prob = predict_email(email_text)
print("Predicted label:", "Spam" if label else "Ham")
print("Probability of spam:", prob)

```

- Supports random selection from stored examples.
- Can also predict via indexing specific email examples.

---

## Example Outputs

**Ham Example:**

```
Email: hi team please find attached the agenda for tomorrow’s meeting
Predicted label: Ham
Probability of spam: 0.12

```

**Spam Example:**

```
Email: claim your free bitcoin today by signing up with this exclusive link
Predicted label: Spam
Probability of spam: 0.998

```

---

## Performance Metrics

**Confusion Matrix (Test Set):**

```
[[1622   33]
 [  17 1700]]

```

**ROC Curve:**

```
ROC-AUC: 0.9988

```

**Interpretation:**

- Accuracy: 0.99, F1-score: 0.99
- ROC-AUC near 1 indicates excellent separation.
- Hybrid features ensure robustness for unstructured emails.

---

## Limitations

- **Domain specificity:** Tuned for noisy, unstructured emails. Will perform poorly on perfectly formatted corporate emails.
- **Embedding computation cost:** Precomputing DistilBERT embeddings is GPU-intensive.
- **Distribution difference:** Real-world inboxes are skewed (more ham than spam), which may require additional calibration for deployment.

---

## Scripts Overview

| Script | Function |
| --- | --- |
| `src/train.py` | Combines TF-IDF, embeddings, and handcrafted features. Trains Logistic Regression. Saves models. |
| `src/precompute_embeddings.py` | Computes and stores DistilBERT embeddings. Supports GPU batching. |
| `src/test.py` | Predicts labels from new email text. Can use random examples or index specific emails. |

---

## Packages & Dependencies

- **scikit-learn:** Logistic Regression, TF-IDF, metrics
- **numpy / pandas:** Data manipulation
- **joblib:** Model/vectorizer persistence
- **transformers (HuggingFace):** DistilBERT embeddings
- **torch:** GPU acceleration for embeddings
- **scipy:** Sparse matrix operations

**Trade-offs:**

- Using TF-IDF alone ignores semantics.
- Using embeddings alone is resource-heavy.
- Combining both yields **accuracy and efficiency**.

---

## Research & References

### **1. Traditional Machine Learning on Spam Emails**

- **Comparative Evaluation of ML Algorithms:**
    
    Logistic Regression, SVM, Random Forest, and Naïve Bayes achieve high accuracy (~95–99%) on multiple text classification/spam corpora, showing classical models are still effective baselines.
    
    🔗 [https://drpress.org/ojs/index.php/HSET/article/view/5805](https://drpress.org/ojs/index.php/HSET/article/view/5805?utm_source=chatgpt.com)
    
- **TF‑IDF + SVM on Enron & TREC Datasets:**
    
    Demonstrates that TF‑IDF with SVM yields high performance on Enron and TREC spam datasets — supporting the use of n‑grams with linear models.
    
    🔗 [https://e-journal.upr.ac.id/index.php/JTI/article/view/22770](https://e-journal.upr.ac.id/index.php/JTI/article/view/22770?utm_source=chatgpt.com)
    
- **Importance of Feature Engineering:**
    
    Shows combining traditional bag‑of‑words (TF‑IDF) with semantic vector embeddings improves classification performance on text tasks.
    
    🔗 [https://link.springer.com/article/10.1186/s43067-024-00151-3](https://link.springer.com/article/10.1186/s43067-024-00151-3?utm_source=chatgpt.com)
    

---

### **2. Transformer & Embedding‑Based Methods**

- **DistilBERT & Transformer Models:**
    
    Details how transformer embeddings (like DistilBERT) capture context and semantics beyond sparse n‑grams.
    
    🔗 [https://www.mdpi.com/2079-9292/14/19/3855](https://www.mdpi.com/2079-9292/14/19/3855?utm_source=chatgpt.com)
    
- **Spam‑T5 and Few‑Shot Models:**
    
    A research paper exploring transformer architectures fine‑tuned or specialized for spam detection.
    
    🔗 [https://arxiv.org/abs/2304.01238](https://arxiv.org/abs/2304.01238?utm_source=chatgpt.com)
    

---

### **3. Nature of Spam Emails & Detection Challenges**

- **Spammer Content Evolution:**
    
    Analyzes how modern spam intentionally obfuscates text to evade filters, validating the need for hybrid models.
    
    🔗 [https://link.springer.com/article/10.1007/s10462-022-10195-4](https://link.springer.com/article/10.1007/s10462-022-10195-4?utm_source=chatgpt.com)
    
- **Header & Metadata Features:**
    
    Shows how header and metadata features (beyond body text) play a significant role in spam detection.
    
    🔗 [https://arxiv.org/abs/2203.10408](https://arxiv.org/abs/2203.10408?utm_source=chatgpt.com)