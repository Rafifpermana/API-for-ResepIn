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
    # Jika data dan tfidf model ada, tapi matrix tidak ada, buat ulang matrix
    if data_valid and tfidf_valid and not matrix_valid:
        print("Data dan TF-IDF model ada, membuat ulang TF-IDF matrix...")
        # Load data dan tfidf model
        data = pd.read_csv(data_path)
        tfidf = joblib.load(tfidf_path)
        
        # Pastikan kolom bahan_bersih ada
        if 'bahan_bersih' not in data.columns:
            print("Membuat kolom bahan_bersih...")
            # Buat kolom bahan_bersih jika tidak ada
            import re
            def bersihkan_teks(teks):
                teks = str(teks)
                teks = re.sub(r'\s+', ' ', teks)
                return teks.strip().lower()
            
            data['ingredients'] = data['ingredients'].fillna('')
            data['bahan_bersih'] = data['ingredients'].apply(bersihkan_teks)
        
        # Buat ulang tf-idf matrix
        print("Membuat TF-IDF matrix...")
        tfidf_matrix = tfidf.transform(data['bahan_bersih'])
        
        # Simpan matrix
        print(f"Menyimpan TF-IDF matrix ({tfidf_matrix.shape})...")
        joblib.dump(tfidf_matrix, matrix_path)
        print("TF-IDF matrix berhasil dibuat ulang dan disimpan")
        
    # Load semua file yang valid
    elif data_valid and tfidf_valid and matrix_valid:
        print("Semua file model valid, memuat model...")
        data = pd.read_csv(data_path)
        tfidf = joblib.load(tfidf_path)
        tfidf_matrix = joblib.load(matrix_path)
        print(f"Berhasil memuat model - Data: {len(data)} resep, Matrix: {tfidf_matrix.shape}")
    
    # Jika beberapa file tidak valid, coba regenerasi dari awal
    else:
        if data_valid:
            print("Hanya data yang valid, membuat ulang model TF-IDF dan matrix...")
            data = pd.read_csv(data_path)
            
            # Pastikan kolom bahan_bersih ada
            if 'bahan_bersih' not in data.columns:
                print("Membuat kolom bahan_bersih...")
                # Buat kolom bahan_bersih jika tidak ada
                import re
                def bersihkan_teks(teks):
                    teks = str(teks)
                    teks = re.sub(r'\s+', ' ', teks)
                    return teks.strip().lower()
                
                data['ingredients'] = data['ingredients'].fillna('')
                data['bahan_bersih'] = data['ingredients'].apply(bersihkan_teks)
            
            # Buat ulang model TF-IDF dan matrix
            print("Membuat model TF-IDF baru...")
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(data['bahan_bersih'])
            
            # Simpan model dan matrix
            print("Menyimpan model TF-IDF dan matrix...")
            joblib.dump(tfidf, tfidf_path)
            joblib.dump(tfidf_matrix, matrix_path)
            print(f"Model berhasil dibuat ulang - Data: {len(data)} resep, Matrix: {tfidf_matrix.shape}")
        else:
            raise Exception("Data tidak valid, tidak bisa membuat model")
            
    # Normalize kategori column
    data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()
    
except Exception as e:
    print(f"Error saat memuat atau membuat model: {e}")
    # Sediakan nilai default untuk testing/deployment
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("Membuat model dummy untuk testing/deployment")
    # Buat data contoh jika file tidak ada
    data = pd.DataFrame({
        'title': ['Resep Contoh'],
        'ingredients': ['bahan contoh'],
        'steps': ['langkah contoh'],
        'kategori_bahan': ['contoh']
    })
    data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()
    
    # Buat model TF-IDF dan matrix dummy
    tfidf = TfidfVectorizer()
    tfidf.fit(['contoh teks'])
    tfidf_matrix = tfidf.transform(['contoh teks'])