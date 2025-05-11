import re
from sklearn.metrics.pairwise import cosine_similarity
from loader import tfidf, tfidf_matrix, data
from utils import to_multiline

# Batasan input
MAX_INGREDIENTS = 20
MAX_QUERY_LENGTH = 200

def filter_by_title_and_ingredients(query: str, top_n: int, page: int):
    """
    Cari resep berdasarkan title & ingredients, kembalikan (results, total_results, total_pages).
    """
    # Buat kata kunci
    keywords = re.sub(r'\s+', ' ', query).strip().lower().split()
    mask_title = data['title'].str.lower().str.contains('|'.join(keywords))
    mask_ing   = data['ingredients'].str.lower().str.contains('|'.join(keywords))
    df = data[mask_title | mask_ing].copy()

    # Similarity score
    q_vec = tfidf.transform([query])
    sim   = cosine_similarity(q_vec, tfidf_matrix).flatten()
    df['score'] = sim[df.index]
    df = df.sort_values('score', ascending=False)

    # Hitung total & halaman
    total_results = len(df)
    total_pages   = (total_results + top_n - 1) // top_n

    # Paginate
    start_idx = page * top_n
    end_idx   = start_idx + top_n
    page_df   = df.iloc[start_idx:end_idx]

    results = [{
        "category":    r['kategori_bahan'],
        "title":       r['title'],
        "ingredients": to_multiline(r['ingredients']),
        "steps":       to_multiline(r['steps']),
    } for _, r in page_df.iterrows()]

    return results, total_results, total_pages

def filter_by_category_and_ingredients(category: str, ingredients: str, top_n: int, page: int):
    """
    Cari resep berdasarkan kategori & daftar bahan, kembalikan (results, total_results, total_pages).
    """
    cat = category.strip().lower()
    df  = data[data['kategori_bahan'] == cat].copy()

    items = [i.strip().lower() for i in ingredients.split(',') if i.strip()]
    for ing in items:
        pat = rf'\b{re.escape(ing)}\b'
        df  = df[df['ingredients'].fillna('').str.lower().str.contains(pat, regex=True)]
        if df.empty:
            break

    total_results = len(df)
    total_pages   = (total_results + top_n - 1) // top_n

    start_idx = page * top_n
    end_idx   = start_idx + top_n
    page_df   = df.iloc[start_idx:end_idx]

    results = [{
        "category":    r['kategori_bahan'],
        "title":       r['title'],
        "ingredients": to_multiline(r['ingredients']),
        "steps":       to_multiline(r['steps']),
    } for _, r in page_df.iterrows()]

    return results, total_results, total_pages