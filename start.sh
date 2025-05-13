#!/bin/bash

# Script to check if the required model files exist
# and download them if they don't

# Create models directory if it doesn't exist
mkdir -p models

# Check if model files exist
if [ ! -f "models/tfidf_model.pkl" ] || [ ! -f "models/tfidf_matrix.pkl" ] || [ ! -f "models/data_resep_bersih.csv" ]; then
    echo "Model files not found. Please make sure to upload them."
    echo "You should have these files in the models directory:"
    echo "- tfidf_model.pkl"
    echo "- tfidf_matrix.pkl"
    echo "- data_resep_bersih.csv"
    
    # Create empty files as placeholders (the app will handle missing files)
    touch models/tfidf_model.pkl
    touch models/tfidf_matrix.pkl
    touch models/data_resep_bersih.csv
fi

# Start the application
exec gunicorn main:app