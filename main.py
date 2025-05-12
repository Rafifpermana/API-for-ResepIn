import os
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

@app.route("/recommend", methods=["POST"])
def recommend_post():
    body      = request.get_json() or {}
    mode      = body.get("mode", "general")
    top_n     = int(body.get("top_n", 5))
    page      = int(body.get("page", 0))

    error         = None
    results       = []
    total_results = 0
    total_pages   = 0

    try:
        if mode == "category_ingredients":
            cat  = body.get("category")
            ings = body.get("ingredients")

            if not cat or not ings:
                error = "Kategori dan bahan wajib diisi"
            elif len(ings.split(",")) > MAX_INGREDIENTS:
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if error:
        return jsonify({"error": error}), 400

    return jsonify({
        "recommendations": results,
        "pagination": {
            "current_page":  page,
            "total_pages":   total_pages,
            "total_results": total_results
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)