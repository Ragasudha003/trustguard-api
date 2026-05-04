import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv("merged_dataset.csv", encoding="latin-1")

print("Original Data:")
print(df.head())

# ==============================
# 2. CLEAN DATA
# ==============================

# Keep only required columns
df = df[['label', 'message']]

# Remove null values
df.dropna(inplace=True)

# Keep only valid labels
df = df[df['label'].isin(['spam', 'ham'])]

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert labels → numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Lowercase text
df['message'] = df['message'].str.lower()

print("\nAfter Cleaning:")
print(df['label'].value_counts())

# ==============================
# 3. TRAIN TEST SPLIT
# ==============================
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. TF-IDF (IMPORTANT)
# ==============================
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),   # improves detection of phrases
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 5. MODEL (BEST CHOICE)
# ==============================
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'   # 🔥 handles imbalance
)

model.fit(X_train_vec, y_train)

# ==============================
# 6. EVALUATION
# ==============================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 7. SAVE MODEL
# ==============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved successfully!")