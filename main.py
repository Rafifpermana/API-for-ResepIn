import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from service import (
    filter_by_title_and_ingredients,
    filter_by_category_and_ingredients,
    MAX_INGREDIENTS,
    MAX_QUERY_LENGTH
)

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Recipe API is running. Use /recommend endpoint for recommendations."

@app.route("/model-status", methods=["GET"])
def model_status():
    """Endpoint untuk memeriksa status model data"""
    from loader import data, tfidf, tfidf_matrix
    
    try:
        return jsonify({
            "status": "ok",
            "data_loaded": len(data) > 0,
            "recipes_count": len(data),
            "tfidf_vocabulary_size": len(tfidf.vocabulary_) if hasattr(tfidf, 'vocabulary_') else 0,
            "tfidf_matrix_shape": tfidf_matrix.shape if hasattr(tfidf_matrix, 'shape') else [0, 0]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/recommend", methods=["POST"])
def recommend_post():
    try:
        body = request.get_json() 
        if body is None:
            return jsonify({"error": "Invalid JSON or missing request body"}), 400
            
        mode = body.get("mode", "general")
        
        # Handle potential non-numeric inputs with defaults
        try:
            top_n = int(body.get("top_n", 5))
            page = int(body.get("page", 0))
        except (ValueError, TypeError):
            top_n = 5
            page = 0

        error = None
        results = []
        total_results = 0
        total_pages = 0

        if mode == "category_ingredients":
            cat = body.get("category")
            ings = body.get("ingredients")

            if not cat or not ings:
                error = "Kategori dan bahan wajib diisi"
            elif ings and len(ings.split(",")) > MAX_INGREDIENTS:
                error = f"Maksimal {MAX_INGREDIENTS} bahan"
            else:
                results, total_results, total_pages = filter_by_category_and_ingredients(
                    cat, ings, top_n, page
                )

        else:  # Mode general
            query = body.get("query")
            if not query:
                error = "Masukkan kata kunci pencarian"
            elif len(query) > MAX_QUERY_LENGTH:
                error = f"Maksimal {MAX_QUERY_LENGTH} karakter"
            else:
                results, total_results, total_pages = filter_by_title_and_ingredients(
                    query, top_n, page
                )

        if error:
            return jsonify({"error": error}), 400

        return jsonify({
            "recommendations": results,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_results": total_results
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in /recommend endpoint: {error_details}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)