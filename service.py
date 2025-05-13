import re
import numpy as np
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
    try:
        # Buat kata kunci
        keywords = re.sub(r'\s+', ' ', query).strip().lower().split()
        if not keywords:
            return [], 0, 0
            
        # Tambahkan error handling jika data kosong
        if len(data) == 0:
            return [], 0, 0
            
        mask_title = data['title'].str.lower().str.contains('|'.join(keywords), regex=True, na=False)
        mask_ing = data['ingredients'].str.lower().str.contains('|'.join(keywords), regex=True, na=False)
        df = data[mask_title | mask_ing].copy()

        # Jika tidak ada hasil yang cocok
        if len(df) == 0:
            return [], 0, 0

        # Similarity score
        q_vec = tfidf.transform([query])
        
        # Jika tfidf_matrix dan q_vec tidak kompatibel, kembalikan hasil berdasarkan filter
        try:
            sim = cosine_similarity(q_vec, tfidf_matrix).flatten()
            valid_indices = [i for i in df.index if i < len(sim)]
            df = df[df.index.isin(valid_indices)].copy()
            df['score'] = sim[df.index]
            df = df.sort_values('score', ascending=False)
        except Exception as e:
            print(f"Error computing similarity: {e}")
            # Fallback to unsorted results
            df['score'] = 1.0

        # Hitung total & halaman
        total_results = len(df)
        total_pages = max(1, (total_results + top_n - 1) // top_n) if total_results > 0 else 0

        # Pastikan page dalam range yang valid
        valid_page = max(0, min(page, total_pages - 1)) if total_pages > 0 else 0

        # Paginate
        start_idx = valid_page * top_n
        end_idx = start_idx + top_n
        page_df = df.iloc[start_idx:end_idx] if start_idx < len(df) else df.iloc[0:0]

        results = []
        for _, r in page_df.iterrows():
            try:
                results.append({
                    "category": r.get('kategori_bahan', ''),
                    "title": r.get('title', ''),
                    "ingredients": to_multiline(r.get('ingredients', '')),
                    "steps": to_multiline(r.get('steps', '')),
                })
            except Exception as e:
                print(f"Error formatting recipe result: {e}")

        return results, total_results, total_pages
        
    except Exception as e:
        print(f"Error in filter_by_title_and_ingredients: {e}")
        return [], 0, 0

def filter_by_category_and_ingredients(category: str, ingredients: str, top_n: int, page: int):
    """
    Cari resep berdasarkan kategori & daftar bahan, kembalikan (results, total_results, total_pages).
    """
    try:
        # Tambahkan error handling jika data kosong
        if len(data) == 0:
            return [], 0, 0
            
        cat = str(category).strip().lower()
        df = data[data['kategori_bahan'] == cat].copy()

        if ingredients:
            items = [i.strip().lower() for i in str(ingredients).split(',') if i.strip()]
            for ing in items:
                try:
                    pat = rf'\b{re.escape(ing)}\b'
                    df = df[df['ingredients'].fillna('').str.lower().str.contains(pat, regex=True)]
                    if df.empty:
                        break
                except Exception as e:
                    print(f"Error filtering by ingredient '{ing}': {e}")
                    continue

        # Hitung total & halaman
        total_results = len(df)
        total_pages = max(1, (total_results + top_n - 1) // top_n) if total_results > 0 else 0

        # Pastikan page dalam range yang valid
        valid_page = max(0, min(page, total_pages - 1)) if total_pages > 0 else 0

        # Paginate
        start_idx = valid_page * top_n
        end_idx = start_idx + top_n
        page_df = df.iloc[start_idx:end_idx] if start_idx < len(df) else df.iloc[0:0]

        results = []
        for _, r in page_df.iterrows():
            try:
                results.append({
                    "category": r.get('kategori_bahan', ''),
                    "title": r.get('title', ''),
                    "ingredients": to_multiline(r.get('ingredients', '')),
                    "steps": to_multiline(r.get('steps', '')),
                })
            except Exception as e:
                print(f"Error formatting recipe result: {e}")

        return results, total_results, total_pages
        
    except Exception as e:
        print(f"Error in filter_by_category_and_ingredients: {e}")
        return [], 0, 0