import re

def to_multiline(raw: str) -> list:
    """
    Split teks resep menjadi list baris/bagian.
    """
    parts = re.split(r'--|\n|\.\s+', str(raw).strip())
    return [p.strip() for p in parts if p.strip()]