import pandas as pd

# Load SMS dataset
sms = pd.read_csv("spam.csv", encoding='latin-1')
sms = sms[['v1', 'v2']]
sms.columns = ['label', 'message']

# Use only SMS dataset
df = sms

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df.to_csv("final_dataset.csv", index=False)

print("✅ Final dataset created:", df.shape)