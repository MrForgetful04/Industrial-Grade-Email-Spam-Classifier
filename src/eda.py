import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/enron_spam_balanced.csv")


print("=== DataFrame Info ===")
print(df.info(), "\n")

print("=== First 5 rows ===")
print(df.head(), "\n")


print("=== Label Distribution ===")
print(df['label'].value_counts())
print("\nPercentages:")
print(df['label'].value_counts(normalize=True) * 100, "\n")


print("=== Missing Values ===")
print(df.isnull().sum(), "\n")


numeric_cols = ['num_urls', 'has_html', 'subject_len', 'body_len', 'num_exclamations', 'num_caps']
print("=== Numeric Feature Summary ===")
print(df[numeric_cols].describe(), "\n")


sns.set_style("whitegrid")

# Label distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Distribution")
plt.xlabel("Label (0=Ham, 1=Spam)")
plt.ylabel("Count")
plt.show()

# Histograms for numeric features
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], bins=30, kde=False)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Boxplots for outliers per label
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x='label', y=col, data=df)
    plt.title(f"{col} by Label (0=Ham, 1=Spam)")
    plt.show()
