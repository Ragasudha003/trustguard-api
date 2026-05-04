import pandas as pd

# ==============================
# FIX COLUMN FORMAT FUNCTION
# ==============================
def fix_columns(df):
    if 'label' in df.columns and 'message' in df.columns:
        return df[['label', 'message']]

    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        return df

    elif 'label' in df.columns and 'text' in df.columns:
        df = df[['label', 'text']]
        df.columns = ['label', 'message']
        return df

    else:
        print("Columns found:", df.columns)
        raise Exception("Unknown format")

# ==============================
# LOAD DATASETS
# ==============================
df1 = pd.read_csv("final_dataset.csv", encoding="latin-1")
df2 = pd.read_csv("spam.csv", encoding="latin-1")
df3 = pd.read_csv("sms_collection.csv", encoding="latin-1")

# ==============================
# FIX FORMAT
# ==============================
df1 = fix_columns(df1)
df2 = fix_columns(df2)
df3 = fix_columns(df3)

# ==============================
# MERGE
# ==============================
df = pd.concat([df1, df2, df3], ignore_index=True)

# ==============================
# CLEAN DATA
# ==============================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['label'].isin(['spam', 'ham'])]

# ==============================
# SAVE
# ==============================
df.to_csv("merged_dataset.csv", index=False)

print("✅ MERGE DONE SUCCESSFULLY!")
print(df['label'].value_counts())