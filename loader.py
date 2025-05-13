import os
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Dapatkan path absolut ke direktori models
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")

# Pastikan direktori models ada
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)

# Path file model
tfidf_path = os.path.join(models_dir, "tfidf_model.pkl")
matrix_path = os.path.join(models_dir, "tfidf_matrix.pkl")
data_path = os.path.join(models_dir, "data_resep_bersih.csv")

# Cek apakah file ada dan valid
tfidf_valid = os.path.exists(tfidf_path) and os.path.getsize(tfidf_path) > 0
matrix_valid = os.path.exists(matrix_path) and os.path.getsize(matrix_path) > 0
data_valid = os.path.exists(data_path) and os.path.getsize(data_path) > 0

print(f"Status file model - TFIDF: {tfidf_valid}, Matrix: {matrix_valid}, Data: {data_valid}")

try:
    # Load data first regardless of other files
    if data_valid:
        print("Memuat dataset resep...")
        data = pd.read_csv(data_path)
        
        # Pastikan kolom bahan_bersih ada dan tidak mengandung NaN
        if 'bahan_bersih' not in data.columns or data['bahan_bersih'].isna().any():
            print("Membuat atau memperbaiki kolom bahan_bersih...")
            # Buat kolom bahan_bersih jika tidak ada atau perbaiki NaN values
            import re
            def bersihkan_teks(teks):
                if pd.isna(teks):
                    return ""  # Return empty string for NaN values
                teks = str(teks)
                teks = re.sub(r'\s+', ' ', teks)
                return teks.strip().lower()
            
            data['ingredients'] = data['ingredients'].fillna('')
            data['bahan_bersih'] = data['ingredients'].apply(bersihkan_teks)
        
        # Jika data dan tfidf model ada, tapi matrix tidak ada, buat ulang matrix
        if tfidf_valid and not matrix_valid:
            print("Data dan TF-IDF model ada, membuat ulang TF-IDF matrix...")
            # Load tfidf model
            tfidf = joblib.load(tfidf_path)
            
            # Buat ulang tf-idf matrix, pastikan tidak ada NaN values
            print("Membuat TF-IDF matrix...")
            # Replace any remaining NaNs with empty string
            clean_text_data = data['bahan_bersih'].fillna('').tolist()
            tfidf_matrix = tfidf.transform(clean_text_data)
            
            # Simpan matrix
            print(f"Menyimpan TF-IDF matrix ({tfidf_matrix.shape})...")
            joblib.dump(tfidf_matrix, matrix_path)
            print("TF-IDF matrix berhasil dibuat ulang dan disimpan")
            
        # Load matrix jika sudah ada
        elif matrix_valid:
            print("Memuat TF-IDF matrix...")
            tfidf = joblib.load(tfidf_path)
            tfidf_matrix = joblib.load(matrix_path)
            print(f"Berhasil memuat model - Data: {len(data)} resep, Matrix: {tfidf_matrix.shape}")
        
        # Jika beberapa file tidak valid, buat model baru dari awal
        else:
            print("Membuat ulang model TF-IDF dan matrix...")
            
            # Buat ulang model TF-IDF dan matrix
            print("Membuat model TF-IDF baru...")
            tfidf = TfidfVectorizer()
            
            # Pastikan tidak ada NaN values
            clean_text_data = data['bahan_bersih'].fillna('').tolist()
            tfidf_matrix = tfidf.fit_transform(clean_text_data)
            
            # Simpan model dan matrix
            print("Menyimpan model TF-IDF dan matrix...")
            joblib.dump(tfidf, tfidf_path)
            joblib.dump(tfidf_matrix, matrix_path)
            print(f"Model berhasil dibuat ulang - Data: {len(data)} resep, Matrix: {tfidf_matrix.shape}")
    else:
        raise Exception("Data tidak valid, tidak bisa membuat model")
            
    # Normalize kategori column if it exists
    if 'kategori_bahan' in data.columns:
        data['kategori_bahan'] = data['kategori_bahan'].fillna('lainnya').str.strip().str.lower()
    else:
        data['kategori_bahan'] = 'lainnya'  # Default kategori
    
except Exception as e:
    print(f"Error saat memuat atau membuat model: {e}")
    # Sediakan nilai default untuk testing/deployment
    print("Membuat model dummy untuk testing/deployment")
    # Buat data contoh jika file tidak ada
    data = pd.DataFrame({
        'title': ['Resep Contoh'],
        'ingredients': ['bahan contoh'],
        'steps': ['langkah contoh'],
        'kategori_bahan': ['contoh'],
        'bahan_bersih': ['bahan contoh']
    })
    data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()
    
    # Buat model TF-IDF dan matrix dummy
    tfidf = TfidfVectorizer()
    tfidf.fit(['bahan contoh'])
    tfidf_matrix = tfidf.transform(['bahan contoh'])