#!/bin/bash

# Script untuk memeriksa apakah file model yang diperlukan ada 
# dan mendownload jika tidak ada

# Buat direktori models jika belum ada
mkdir -p models

# Fungsi untuk memeriksa apakah file valid (tidak kosong)
check_file() {
    if [ ! -f "$1" ] || [ ! -s "$1" ]; then
        return 1
    fi
    return 0
}

# Periksa apakah file model ada dan valid
if ! check_file "models/tfidf_model.pkl" || 
   ! check_file "models/tfidf_matrix.pkl" || 
   ! check_file "models/data_resep_bersih.csv"; then
    
    echo "File model tidak ditemukan atau kosong."
    
    # Jika MODEL_URL tersedia, unduh model dari sana
    if [ -n "$MODEL_URL" ]; then
        echo "Mencoba mengunduh model dari $MODEL_URL"
        
        # Buat direktori temporary untuk download
        mkdir -p temp_download
        
        # Download file model
        if curl -L "$MODEL_URL/tfidf_model.pkl" -o temp_download/tfidf_model.pkl && 
           curl -L "$MODEL_URL/tfidf_matrix.pkl" -o temp_download/tfidf_matrix.pkl &&
           curl -L "$MODEL_URL/data_resep_bersih.csv" -o temp_download/data_resep_bersih.csv; then
            
            # Pindahkan file yang berhasil diunduh ke direktori models
            mv temp_download/* models/
            echo "Model berhasil diunduh"
        else
            echo "Gagal mengunduh model dari $MODEL_URL"
        fi
        
        # Hapus direktori temporary
        rm -rf temp_download
    else
        echo "MODEL_URL tidak ditentukan. Tidak bisa mengunduh model."
        echo "Aplikasi akan berjalan dengan model dummy."
        
        # Buat file kosong sebagai placeholder (aplikasi akan menangani file yang hilang)
        touch models/tfidf_model.pkl
        touch models/tfidf_matrix.pkl
        touch models/data_resep_bersih.csv
    fi
fi

# Tampilkan informasi file model
echo "Informasi file model:"
echo "- tfidf_model.pkl: $(ls -la models/tfidf_model.pkl)"
echo "- tfidf_matrix.pkl: $(ls -la models/tfidf_matrix.pkl)"
echo "- data_resep_bersih.csv: $(ls -la models/data_resep_bersih.csv)"

# Jalankan aplikasi
exec gunicorn main:app