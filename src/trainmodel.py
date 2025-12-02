import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os


# ---------------------------------------------------------
# UČITAVANJE PODATAKA
# ---------------------------------------------------------
df = pd.read_csv("data/products.csv")

# Uklanjanje svih redova sa NaN vrednostima
df = df.dropna()

# Čišćenje naziva kolona od razmaka
df.columns = df.columns.str.strip()

# Preimenovanje ako postoji "_Product Code"
if "_Product Code" in df.columns:
    df.rename(columns={"_Product Code": "Product Code"}, inplace=True)

# Pretvaranje datuma u datetime format
df["Listing Date"] = pd.to_datetime(df["Listing Date"], format="%m/%d/%Y", errors="coerce")

# Ponovno uklanjanje NaN vrednosti posle parsiranja
df = df.dropna()

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
# Čišćenje teksta
df["clean_title"] = (
    df["Product Title"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
    .str.strip()
)

# Numeric feature-i
df["title_length"] = df["clean_title"].str.len()
df["word_count"] = df["clean_title"].str.split().str.len()
df["digit_count"] = df["clean_title"].str.count(r"\d")

# ---------------------------------------------------------
# DEFINISANJE FEATURE-A I LABLE-A
# ---------------------------------------------------------
X = df[["clean_title", "title_length", "word_count", "digit_count"]]
y = df["Category Label"]

# ---------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# COLUMN TRANSFORMER (TF-IDF + SCALER)
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(ngram_range=(1,2), max_features=50000), "clean_title"),
        ("nums", MinMaxScaler(), ["title_length", "word_count", "digit_count"])
    ]
)

# ---------------------------------------------------------
# PIPELINE (PREPROCESSING + MODEL)
# ---------------------------------------------------------
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier())
])

# ---------------------------------------------------------
# TRENING MODELA
# ---------------------------------------------------------
print("\nTreniram model...")
pipeline.fit(X_train, y_train)
print("Model uspešno treniran.\n")

# ---------------------------------------------------------
# ČUVANJE MODELA
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/product_classifier.pkl")

print("Model je sačuvan kao: models/product_classifier.pkl")
