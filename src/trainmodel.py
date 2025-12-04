# Importujemo biblioteke neophodne da bi uradili zadatak
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Razliciti modeli
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Ucitavamo podatke
df = pd.read_csv("data/products.csv")

# Uklanjanje NaN vrednosti
df = df.dropna()

# Ciscenje naziva kolona od razmaka
df.columns = df.columns.str.strip()

# Preimenovanje "_Product Code"
if "_Product Code" in df.columns:
    df.rename(columns={"_Product Code": "Product Code"}, inplace=True)

# Listing Date u datetime
df["Listing Date"] = pd.to_datetime(df["Listing Date"], format="%m/%d/%Y", errors="coerce")

# Ponovno uklanjanje nan
df = df.dropna()


# Radimo feature engineering, kolonu "Product Title" prepravljamo da postane "clean_title"
df["clean_title"] = (
    df["Product Title"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
    .str.strip()
)

# Pravimo nove metrike poput velicine teksta, broja reci i broj karaktera radi boljeg uvidjaja 
df["title_length"] = df["clean_title"].str.len()
df["word_count"] = df["clean_title"].str.split().str.len()
df["digit_count"] = df["clean_title"].str.count(r"\d")


# Definisemo x i y
X = df[["clean_title", "title_length", "word_count", "digit_count"]]
y = df["Category Label"]


#Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(ngram_range=(1,2), max_features=50000), "clean_title"),
        ("nums", MinMaxScaler(), ["title_length", "word_count", "digit_count"])
    ]
)


# Ovde su modeli koje cemo koristiti za poredjenje
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "Linear SVM": LinearSVC()
}


best_model = None
best_acc = 0
best_name = ""


# Treniranje i evaulacija
print("\n=== Training models ===\n")

for name, model in models.items():
    print(f"Training: {name} ...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} accuracy: {acc:.4f}\n")

    # Cuvamo najbolji model
    if acc > best_acc:
        best_acc = acc
        best_model = pipeline
        best_name = name


#Cuvanje najboljeg modela
joblib.dump(best_model, "model/product_classifier.pkl")

print(f"Best model: {best_name}  (accuracy: {best_acc:.4f})")
print("Model saved to: model/product_classifier.pkl")
