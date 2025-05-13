import re

def to_multiline(raw: str) -> list:
    """
    Split teks resep menjadi list baris/bagian.
    """
    try:
        if not raw or not isinstance(raw, str):
            return []
            
        parts = re.split(r'--|\n|\.\s+', str(raw).strip())
        return [p.strip() for p in parts if p.strip()]
    except Exception as e:
        print(f"Error in to_multiline: {e}")
        return [str(raw)] if raw else []