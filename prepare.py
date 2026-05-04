import pandas as pd

# 1️⃣ Load SMS spam dataset (Kaggle)
sms = pd.read_csv("spam.csv", encoding='latin-1')

# Clean columns
sms = sms[['v1', 'v2']]
sms.columns = ['label', 'message']

# Convert labels
sms['label'] = sms['label'].map({'spam': 'spam', 'ham': 'ham'})

# 2️⃣ Load phishing dataset
phish = pd.read_csv("Phishing_Legitimate_full.csv")

# Assume column names (adjust if needed)
phish['label'] = phish['CLASS_LABEL'].map({1: 'spam', 0: 'ham'})
phish['message'] = phish['url']  # or text column if exists

phish = phish[['label', 'message']]

# 3️⃣ Load generated dataset (optional)
gen = pd.read_csv("generated_dataset.csv")

# 4️⃣ Combine all
df = pd.concat([sms, phish, gen])

# 5️⃣ Clean data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 6️⃣ Save final dataset
df.to_csv("final_dataset.csv", index=False)

print("✅ Final dataset created:", df.shape)