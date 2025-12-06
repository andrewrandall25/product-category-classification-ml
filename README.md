# Product Category Classification (Machine Learning Project)

Ovaj projekat automatski klasifikuje proizvode u odgovarajuće kategorije na osnovu njihovog naziva (*Product Title*).  
Model je treniran nad realnim skupom podataka i koristi kombinaciju TF-IDF vektora i numerickih feature-a.

---

## Struktura +

project/
│
├── data/
│ └── products.csv
│
├── model/
│ └── product_classifier.pkl
│
├── notebooks/
│ └── data_analysis.ipynb
│
├── src/
│ ├── trainmodel.py
│ └── predict_category.py
│ └── .gitignore
├── requirements.txt
└── README.md


---

## Instalacija okruženja

1. Klonirajte repozitorijum:

git clone <URL_REPOZITORIJUMA>
cd <ime_projekta>


2. Kreirajte virtualno okruženje:

python -m venv venv


3. Aktivirajte ga:

Windows:

venv\Scripts\activate

macOS / Linux:


4. Instalirajte requirements:


---

## Treniranje modela

Ako zelite da ponovo istrenirate model:

python src/train_model.py


Po zavrsetku treniranja, model će biti sacuvan u:


model/product_classifier.pkl


---

## Predikcija kategorije (interaktivni rezim)

Pokrenite interaktivni skript:


python src/predict_category.py


Primer rada:

Enter product title: iphone 7 32gb gold
Predicted category: Mobile Phones


Za izlazak iz programa:

exit


---

## Notebook analiza

Kompletna analiza, EDA, feature engineering i testiranje modela nalaze se u:

notebooks/01_data_analysis_and_modeling.ipynb


Notebook je detaljno dokumentovan i koristi se za razvoj modela.

---

##  Koriscene tehnologije

- Python  
- Pandas  
- Scikit-learn  
- TF-IDF vektorizacija  
- LinearSVC / RandomForest (modeli)  
- Joblib (cuvanje modela)  
- Jupyter/Colab za analizu  

---

## Andrew Randall

Ovaj projekat razvijen je u okviru kursnog zadatka i predstavlja kompletnu ML pipelines strukturu spremnu za dalji razvoj i timski rad.

---




