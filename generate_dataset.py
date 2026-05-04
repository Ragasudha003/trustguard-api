import pandas as pd
import random

spam = ["win lottery", "click link", "update kyc now"]
ham = ["hello", "how are you", "meeting at 5"]

data = []

for _ in range(10000):
    data.append(["spam", random.choice(spam)])

for _ in range(10000):
    data.append(["ham", random.choice(ham)])

df = pd.DataFrame(data, columns=["label", "message"])
df.to_csv("generated_dataset.csv", index=False)

print("✅ generated_dataset.csv created")