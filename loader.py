import os
import pandas as pd
import joblib

# Get absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")

# Make sure models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)

try:
    # Load model dan data dengan format terkompresi
    tfidf_path = os.path.join(models_dir, "tfidf_model.pkl")
    matrix_path = os.path.join(models_dir, "tfidf_matrix.pkl")
    data_path = os.path.join(models_dir, "data_resep_bersih.csv")
    
    tfidf = joblib.load(tfidf_path)  # Model TF-IDF terkompresi
    tfidf_matrix = joblib.load(matrix_path)  # Changed to load with joblib
    data = pd.read_csv(data_path)  # Changed to read CSV

    # Normalisasi kolom kategori
    data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()
    
except Exception as e:
    print(f"Error loading models or data: {e}")
    # Provide default values for testing/deployment
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create empty/sample data if files don't exist
    data = pd.DataFrame({
        'title': ['Sample Recipe'],
        'ingredients': ['sample ingredients'],
        'steps': ['sample steps'],
        'kategori_bahan': ['sample']
    })
    data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()
    
    # Create dummy TF-IDF model and matrix
    tfidf = TfidfVectorizer()
    tfidf.fit(['sample text'])
    tfidf_matrix = tfidf.transform(['sample text'])