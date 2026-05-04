import pandas as pd

# Read raw file (tab-separated)
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Save as CSV
df.to_csv("sms_collection.csv", index=False)

print("✅ Converted to CSV successfully!")
print(df['label'].value_counts())