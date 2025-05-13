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

# Fungsi untuk memeriksa file
def check_files():
    tfidf_path = os.path.join(models_dir, "tfidf_model.pkl")
    matrix_path = os.path.join(models_dir, "tfidf_matrix.pkl")
    data_path = os.path.join(models_dir, "data_resep_bersih.csv")
    
    tfidf_ok = os.path.exists(tfidf_path) and os.path.getsize(tfidf_path) > 0
    matrix_ok = os.path.exists(matrix_path) and os.path.getsize(matrix_path) > 0
    data_ok = os.path.exists(data_path) and os.path.getsize(data_path) > 0
    
    print(f"Status file model - TFIDF: {tfidf_ok}, Matrix: {matrix_ok}, Data: {data_ok}")
    
    return tfidf_ok, matrix_ok, data_ok, tfidf_path, matrix_path, data_path

# Coba load model
try:
    # Cek status file
    tfidf_ok, matrix_ok, data_ok, tfidf_path, matrix_path, data_path = check_files()
    
    # Gunakan model dari file jika semua file valid
    if tfidf_ok and matrix_ok and data_ok:
        print("Memuat model dari file lokal...")
        tfidf = joblib.load(tfidf_path)
        tfidf_matrix = joblib.load(matrix_path)
        data = pd.read_csv(data_path)
        print(f"Berhasil memuat {len(data)} resep dari file lokal")
    
    # Jika tidak, coba download dari URL
    else:
        model_url = os.environ.get("MODEL_URL")
        if model_url:
            print(f"Mengunduh model dari {model_url}...")
            
            # Unduh dan simpan
            try:
                if not tfidf_ok:
                    print(f"Mengunduh tfidf_model.pkl dari {model_url}/tfidf_model.pkl")
                    tfidf_resp = requests.get(f"{model_url}/tfidf_model.pkl", timeout=60)
                    if tfidf_resp.status_code == 200:
                        with open(tfidf_path, 'wb') as f:
                            f.write(tfidf_resp.content)
                        print("Berhasil mengunduh tfidf_model.pkl")
                
                if not matrix_ok:
                    print(f"Mengunduh tfidf_matrix.pkl dari {model_url}/tfidf_matrix.pkl")
                    matrix_resp = requests.get(f"{model_url}/tfidf_matrix.pkl", timeout=60)
                    if matrix_resp.status_code == 200:
                        with open(matrix_path, 'wb') as f:
                            f.write(matrix_resp.content)
                        print("Berhasil mengunduh tfidf_matrix.pkl")
                
                if not data_ok:
                    print(f"Mengunduh data_resep_bersih.csv dari {model_url}/data_resep_bersih.csv")
                    data_resp = requests.get(f"{model_url}/data_resep_bersih.csv", timeout=60)
                    if data_resp.status_code == 200:
                        with open(data_path, 'wb') as f:
                            f.write(data_resp.content)
                        print("Berhasil mengunduh data_resep_bersih.csv")
                
                # Periksa kembali status file setelah download
                tfidf_ok, matrix_ok, data_ok, _, _, _ = check_files()
                
                if tfidf_ok and matrix_ok and data_ok:
                    print("Semua file berhasil diunduh, memuat model...")
                    tfidf = joblib.load(tfidf_path)
                    tfidf_matrix = joblib.load(matrix_path)
                    data = pd.read_csv(data_path)
                    print(f"Berhasil memuat {len(data)} resep dari file yang diunduh")
                else:
                    raise Exception("Beberapa file masih tidak valid setelah download")
                    
            except Exception as e:
                print(f"Error saat mengunduh model: {e}")
                raise
        else:
            print("URL model tidak ditemukan di environment variables")
            raise Exception("File model tidak valid dan URL tidak dikonfigurasi")
    
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