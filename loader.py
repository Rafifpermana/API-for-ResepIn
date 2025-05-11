import pandas as pd
import joblib
from scipy.sparse import load_npz

# Load model dan data dengan format terkompresi
tfidf = joblib.load("models/tfidf_model.pkl")  # Model TF-IDF terkompresi
tfidf_matrix = load_npz("models/tfidf_matrix.npz")  # Format sparse matrix
data = pd.read_parquet("models/data_resep_bersih.parquet")  # Format Parquet

# Normalisasi kolom kategori
data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()