from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from service import (
    filter_by_title_and_ingredients,
    filter_by_category_and_ingredients,
    MAX_INGREDIENTS,
    MAX_QUERY_LENGTH
)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema untuk input POST
class RecommendRequest(BaseModel):
    mode: str = "general"
    top_n: int = 5
    page: int = 0
    query: str | None = None
    category: str | None = None
    ingredients: str | None = None

@app.post("/recommend")
async def recommend_post(body: RecommendRequest):
    mode = body.mode
    top_n = body.top_n
    page = body.page

    error = None
    results = []
    total_results = 0
    total_pages = 0

    try:
        if mode == "category_ingredients":
            cat = body.category
            ings = body.ingredients

            if not cat or not ings:
                error = "Kategori dan bahan wajib diisi"
            elif len(ings.split(",")) > MAX_INGREDIENTS:
                error = f"Maksimal {MAX_INGREDIENTS} bahan"
            else:
                results, total_results, total_pages = filter_by_category_and_ingredients(
                    cat, ings, top_n, page
                )
        else:
            query = body.query
            if not query:
                error = "Masukkan kata kunci pencarian"
            elif len(query) > MAX_QUERY_LENGTH:
                error = f"Maksimal {MAX_QUERY_LENGTH} karakter"
            else:
                results, total_results, total_pages = filter_by_title_and_ingredients(
                    query, top_n, page
                )

    except Exception as e:
        return {"error": str(e)}

    if error:
        return {"error": error}

    return {
        "recommendations": results,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_results": total_results
        }
    }
