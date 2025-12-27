import pandas as pd

# Load full cleaned dataset
df = pd.read_csv("data/enron_spam_cleaned.csv")

# Load balanced dataset
balanced = pd.read_csv("data/enron_spam_balanced.csv")

# Emails not included in balanced set
excluded = df.drop(balanced.index, errors='ignore')

print(f"Excluded emails available for testing: {len(excluded)}")

# ----------------------------
# Interactive loop
# ----------------------------
while True:
    inp = input("Press Enter to see a new excluded email (or type 'q' to quit): ")
    if inp.lower() == 'q':
        break

    sample = excluded.sample(n=1)  # new random sample each time
    row = sample.iloc[0]

    print("\nIndex:", row.name)
    print("Label:", "Spam" if row['label'] == 1 else "Ham")
    print("Subject Clean:", row['subject_clean'])
    print("Body Clean:", row['body_clean'])
    print("Full Text:\n", row['text'])
    print("-"*80)

