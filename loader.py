import pandas as pd
import joblib


# Load model dan data dengan format terkompresi
tfidf = joblib.load("models/tfidf_model.pkl")  # Model TF-IDF terkompresi
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")  # Changed to load with joblib
data = pd.read_csv("models/data_resep_bersih.csv")  # Changed to read CSV

# Normalisasi kolom kategori
data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()
