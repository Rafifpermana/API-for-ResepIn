import os
import pandas as pd
import joblib
import requests
import io

# Dapatkan path absolut ke direktori models
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")

# Pastikan direktori models ada
if not os.path.exists(models_dir):
    os.makedirs(models_dir, exist_ok=True)

# Fungsi untuk memeriksa apakah file valid
def is_valid_file(file_path):
    if not os.path.exists(file_path):
        return False
    # Cek ukuran file (jika 0 berarti kosong)
    return os.path.getsize(file_path) > 0

# Coba load model dari file lokal atau URL
try:
    # Path file lokal
    tfidf_path = os.path.join(models_dir, "tfidf_model.pkl")
    matrix_path = os.path.join(models_dir, "tfidf_matrix.pkl")
    data_path = os.path.join(models_dir, "data_resep_bersih.csv")
    
    # Cek apakah file ada dan tidak kosong
    valid_tfidf = is_valid_file(tfidf_path)
    valid_matrix = is_valid_file(matrix_path)
    valid_data = is_valid_file(data_path)
    
    print(f"Status file model - TFIDF: {valid_tfidf}, Matrix: {valid_matrix}, Data: {valid_data}")
    
    # Load model jika file valid
    if valid_tfidf and valid_matrix and valid_data:
        print("Memuat model dari file lokal...")
        tfidf = joblib.load(tfidf_path)
        tfidf_matrix = joblib.load(matrix_path)
        data = pd.read_csv(data_path)
        print(f"Berhasil memuat {len(data)} resep")
    else:
        # Coba ambil dari URL jika ada
        model_url = os.environ.get("MODEL_URL")
        if model_url:
            print(f"Mengunduh model dari {model_url}...")
            # Contoh: MODEL_URL=https://example.com/models/
            try:
                # Download dan load langsung dari URL
                tfidf_url = f"{model_url}/tfidf_model.pkl"
                matrix_url = f"{model_url}/tfidf_matrix.pkl"
                data_url = f"{model_url}/data_resep_bersih.csv"
                
                # Download tfidf model
                tfidf_response = requests.get(tfidf_url)
                tfidf = joblib.load(io.BytesIO(tfidf_response.content))
                
                # Download matrix
                matrix_response = requests.get(matrix_url)
                tfidf_matrix = joblib.load(io.BytesIO(matrix_response.content))
                
                # Download data
                data_response = requests.get(data_url)
                data = pd.read_csv(io.StringIO(data_response.text))
                
                print(f"Berhasil mengunduh dan memuat {len(data)} resep")
                
                # Simpan file untuk penggunaan berikutnya
                joblib.dump(tfidf, tfidf_path)
                joblib.dump(tfidf_matrix, matrix_path)
                data.to_csv(data_path, index=False)
            except Exception as e:
                print(f"Gagal mengunduh model: {e}")
                raise
        else:
            print("URL model tidak ditemukan di environment variables")
            raise FileNotFoundError("File model tidak valid dan URL tidak dikonfigurasi")
    
    # Normalisasi kolom kategori
    data['kategori_bahan'] = data['kategori_bahan'].str.strip().str.lower()
    
except Exception as e:
    print(f"Error saat memuat model atau data: {e}")
    # Sediakan nilai default untuk testing/deployment
    import numpy as np
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