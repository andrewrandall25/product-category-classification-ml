import joblib
import pandas as pd
import re

# Ucitavamo model
model = joblib.load("model/product_classifier.pkl")

print("Model loaded successfully!")
print("Type 'exit' to stop.\n")

# Ciscenje teksta na isti nacin kao u train_model.py."""
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return text.strip()

while True:
    title = input("Enter product title: ")

    if title.lower() == "exit":
        print("Exiting...")
        break

    # Sredjujemo tekst kao i u trainmodelu
    clean_title = clean_text(title)

    # Nove metrike
    title_length = len(clean_title)
    word_count = len(clean_title.split())
    digit_count = sum(char.isdigit() for char in clean_title)

    # Pravimo DataFrame za predikciju
    user_input = pd.DataFrame([{
        "clean_title": clean_title,
        "title_length": title_length,
        "word_count": word_count,
        "digit_count": digit_count
    }])

    # Predvidjanje kategorije
    prediction = model.predict(user_input)[0]

    print(f"\nPredicted category: {prediction}")
    print("-" * 40 + "\n")
