from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import ast
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import html as html_lib
import traceback
from dotenv import load_dotenv
import os
from groq import Groq
import httpx
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import csv
from pathlib import Path

# Load environment variables
load_dotenv()

# Google Gemini API for intelligent menu extraction
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

app = FastAPI(title="Swaad Recipe Recommendation API")

_embedding_model: Optional[SentenceTransformer] = None
_pinecone_index = None
_groq_client = None

@app.on_event("startup")
def preload_models():
    """Preload sentence-transformer model on startup to avoid first-request delay."""
    global _embedding_model, _groq_client
    if _embedding_model is None:
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        print(f"Preloading sentence-transformer model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        print("Sentence-transformer model loaded.")
    if _groq_client is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            _groq_client = Groq(api_key=groq_api_key)
            print("Groq client initialized.")

_ingredient_flavor_map = None
_ingredient_upsert_done = False
_taste_infer_cache: Dict[str, List[float]] = {}

USER_METADATA_MAP = {
    "default": {
        "location": os.getenv("DEFAULT_USER_LOCATION", ""),
        "allergies": ["Nuts", "Cabbage", "Cauliflower"],
        "favorite_dishes": [
            {"name": "Paneer butter masala", "category": "mains"},
            {"name": "Veg Manchurian", "category": "appetizer"},
            {"name": "Peppy Paneer Pizza", "category": "mains"},
        ],
        "diet_type": "veg",
        "flavor_profile": {
            "appetizer": {"spicy": 0.6, "sweet": 0.2, "umami": 0.7, "sour": 0.3, "salty": 0.6},
            "mains": {"spicy": 0.7, "sweet": 0.2, "umami": 0.8, "sour": 0.3, "salty": 0.7},
            "desserts": {"spicy": 0.1, "sweet": 0.8, "umami": 0.2, "sour": 0.2, "salty": 0.2},
            "allergies": ["Nuts", "Cabbage", "Cauliflower"],
            "favorite_dishes": [
                {"name": "Paneer butter masala", "category": "mains"},
                {"name": "Veg Manchurian", "category": "appetizer"},
                {"name": "Peppy Paneer Pizza", "category": "mains"},
            ],
            "diet_type": "veg",
        },
        "pending_query": None,
    }
    ,
    "dummy2": {
        "location": "New York, NY",
        "allergies": ["shellfish"],
        "favorite_dishes": [
            {"name": "Pepperoni Pizza", "category": "mains"},
            {"name": "Chicken Wings", "category": "appetizer"},
            {"name": "Cheesecake", "category": "desserts"}
        ],
        "diet_type": "non-veg",
        "flavor_profile": {
            "appetizer": {"spicy": 0.8, "sweet": 0.1, "umami": 0.7, "sour": 0.2, "salty": 0.8},
            "mains": {"spicy": 0.6, "sweet": 0.2, "umami": 0.8, "sour": 0.2, "salty": 0.8},
            "desserts": {"spicy": 0.1, "sweet": 0.9, "umami": 0.2, "sour": 0.2, "salty": 0.2},
            "allergies": ["shellfish"],
            "favorite_dishes": [
                {"name": "Pepperoni Pizza", "category": "mains"},
                {"name": "Chicken Wings", "category": "appetizer"},
                {"name": "Cheesecake", "category": "desserts"}
            ],
            "diet_type": "non-veg",
        },
        "pending_query": None,
    }
    ,
    "dummy3": {
        "location": "San Francisco, CA",
        "allergies": ["gluten"],
        "favorite_dishes": [
            {"name": "Spicy Tuna Roll", "category": "mains"},
            {"name": "Tonkotsu Ramen", "category": "mains"},
            {"name": "Mango Mochi", "category": "desserts"}
        ],
        "diet_type": "mix",
        "flavor_profile": {
            "appetizer": {"spicy": 0.5, "sweet": 0.2, "umami": 0.7, "sour": 0.4, "salty": 0.6},
            "mains": {"spicy": 0.7, "sweet": 0.2, "umami": 0.9, "sour": 0.3, "salty": 0.7},
            "desserts": {"spicy": 0.1, "sweet": 0.8, "umami": 0.3, "sour": 0.3, "salty": 0.2},
            "allergies": ["gluten"],
            "favorite_dishes": [
                {"name": "Spicy Tuna Roll", "category": "mains"},
                {"name": "Tonkotsu Ramen", "category": "mains"},
                {"name": "Mango Mochi", "category": "desserts"}
            ],
            "diet_type": "mix",
        },
        "pending_query": None,
    }
}

def _normalize_favorite_dishes(favorite_dishes: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not favorite_dishes or not isinstance(favorite_dishes, list):
        return out
    for d in favorite_dishes:
        if isinstance(d, dict):
            name = (d.get("name") or d.get("dish") or "").strip()
            if not name:
                continue
            category = (d.get("category") or "mains").strip() or "mains"
            out.append({"name": name, "category": category})
        else:
            name = str(d).strip()
            if not name:
                continue
            out.append({"name": name, "category": "mains"})
    return out

def _get_dummy_user(user_key: str = "default") -> Dict[str, Any]:
    key = (user_key or "default").strip() or "default"
    u = USER_METADATA_MAP.get(key)
    if not isinstance(u, dict):
        USER_METADATA_MAP[key] = {
            "location": os.getenv("DEFAULT_USER_LOCATION", ""),
            "allergies": [],
            "favorite_dishes": [],
            "diet_type": "mix",
            "flavor_profile": None,
            "pending_query": None,
        }
        u = USER_METADATA_MAP[key]
    if "favorite_dishes" in u:
        u["favorite_dishes"] = _normalize_favorite_dishes(u.get("favorite_dishes"))
    return u

def _dummy_user_to_user_profile(u: Dict[str, Any]) -> Optional["UserProfile"]:
    if not u or not isinstance(u, dict):
        return None
    fp = u.get("flavor_profile")
    if not fp or not isinstance(fp, dict):
        return None
    try:
        merged = dict(fp)
        if "allergies" not in merged:
            merged["allergies"] = u.get("allergies") or []
        if "favorite_dishes" not in merged:
            merged["favorite_dishes"] = u.get("favorite_dishes") or []
        if "diet_type" not in merged:
            merged["diet_type"] = u.get("diet_type") or "mix"
        return UserProfile(**merged)
    except Exception:
        return None

def _sync_dummy_user_from_request(request: Any) -> None:
    key = getattr(request, "user_key", None) or "default"
    u = _get_dummy_user(key)
    try:
        if getattr(request, "location", None):
            u["location"] = getattr(request, "location")
    except Exception:
        pass
    try:
        if getattr(request, "diet_type", None):
            u["diet_type"] = getattr(request, "diet_type")
    except Exception:
        pass
    try:
        if getattr(request, "favorite_dishes", None):
            fav = getattr(request, "favorite_dishes")
            if isinstance(fav, list):
                raw = [d.model_dump() if hasattr(d, "model_dump") else d for d in fav]
                u["favorite_dishes"] = _normalize_favorite_dishes(raw)
    except Exception:
        pass
    try:
        if getattr(request, "user_profile", None):
            up = getattr(request, "user_profile")
            if hasattr(up, "model_dump"):
                up_dict = up.model_dump()
            elif isinstance(up, dict):
                up_dict = up
            else:
                up_dict = None
            if isinstance(up_dict, dict):
                u["flavor_profile"] = {
                    "appetizer": up_dict.get("appetizer"),
                    "mains": up_dict.get("mains"),
                    "desserts": up_dict.get("desserts"),
                    "allergies": up_dict.get("allergies", u.get("allergies") or []),
                    "favorite_dishes": up_dict.get("favorite_dishes", u.get("favorite_dishes") or []),
                    "diet_type": up_dict.get("diet_type", u.get("diet_type") or "mix"),
                }
                if isinstance(up_dict.get("allergies"), list):
                    u["allergies"] = up_dict.get("allergies")
                if isinstance(up_dict.get("favorite_dishes"), list):
                    u["favorite_dishes"] = _normalize_favorite_dishes(up_dict.get("favorite_dishes"))
                if up_dict.get("diet_type"):
                    u["diet_type"] = up_dict.get("diet_type")
    except Exception:
        pass

_dish_classification_cache: Dict[str, str] = {}

def _merge_unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items or []:
        if not x or not isinstance(x, str):
            continue
        v = x.strip()
        if not v:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out

def _classify_dish_with_groq(dish_name: str) -> str:
    """
    Classify a dish as 'veg' or 'non-veg' using Groq LLM.
    Returns: 'veg' or 'non-veg'
    Uses caching to avoid repeated API calls.
    """
    if not dish_name or not isinstance(dish_name, str):
        return "veg"  # Default to veg if invalid
    
    cache_key = dish_name.lower().strip()
    if cache_key in _dish_classification_cache:
        return _dish_classification_cache[cache_key]
    
    try:
        groq_client = _get_groq_client()
        prompt = f"""Classify this dish as either 'veg' or 'non-veg'.

Dish: {dish_name}

Rules:
- 'non-veg' includes: meat, poultry, fish, seafood, eggs, and any animal products
- 'veg' includes: vegetables, fruits, dairy, grains, legumes, plant-based items
- If unclear or dish name doesn't specify, default to 'veg'

Respond with ONLY the word 'veg' or 'non-veg', nothing else."""
        
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=10
        )
        
        result = completion.choices[0].message.content.strip().lower()
        classification = "non-veg" if "non" in result else "veg"
        
        _dish_classification_cache[cache_key] = classification
        return classification
        
    except Exception as e:
        print(f"[WARNING] Groq classification failed for '{dish_name}': {e}")
        # Fallback to basic keyword check
        t = dish_name.lower()
        nonveg_keywords = {"chicken", "beef", "pork", "bacon", "ham", "turkey", "lamb", 
                          "mutton", "duck", "fish", "salmon", "tuna", "shrimp", "prawn", 
                          "crab", "lobster", "egg", "meat", "seafood"}
        classification = "non-veg" if any(k in t for k in nonveg_keywords) else "veg"
        _dish_classification_cache[cache_key] = classification
        return classification

def _is_nonveg_text(text: str) -> bool:
    """Check if dish is non-veg using Groq classification."""
    return _classify_dish_with_groq(text) == "non-veg"

def _filter_dishes_by_diet(dishes: List[str], diet_type: Optional[str]) -> List[str]:
    """Filter dishes based on diet type using Groq LLM classification."""
    d = (diet_type or "mix").strip().lower()
    if d not in {"veg", "vegetarian", "non-veg", "nonveg", "mix"}:
        d = "mix"
    if d == "mix":
        return dishes or []
    
    filtered = []
    for dish in (dishes or []):
        if not dish or not isinstance(dish, str):
            continue
        classification = _classify_dish_with_groq(dish)
        
        if d in {"veg", "vegetarian"}:
            # For veg users, only include veg dishes
            if classification == "veg":
                filtered.append(dish)
        else:
            # For non-veg users, only include non-veg dishes
            if classification == "non-veg":
                filtered.append(dish)
    
    return filtered

def _extract_text_from_html_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    try:
        s = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    s = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", s)
    s = re.sub(r"(?i)<br\\s*/?>", "\n", s)
    s = re.sub(r"(?i)</(p|li|div|tr|h1|h2|h3|h4|h5|h6)>", "\n", s)
    s = re.sub(r"(?s)<.*?>", " ", s)
    s = html_lib.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()

async def _fetch_menu_url(menu_url: str) -> Optional[httpx.Response]:
    if not menu_url:
        return None
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as http_client:
            resp = await http_client.get(menu_url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code >= 400:
                return None
            return resp
    except Exception:
        return None

async def _menu_url_to_dishes(menu_url: str) -> List[str]:
    resp = await _fetch_menu_url(menu_url)
    if resp is None:
        return []

    content_type = (resp.headers.get("content-type") or "").lower()
    url_lc = (menu_url or "").lower()

    is_pdf = "application/pdf" in content_type or url_lc.endswith(".pdf")
    is_img = content_type.startswith("image/") or any(url_lc.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) 
    is_html = "text/html" in content_type or "application/xhtml" in content_type or url_lc.endswith(".html") or url_lc.endswith(".htm")

    raw = resp.content
    if not raw:
        return []

    if is_pdf:
        try:
            if not os.getenv("GEMINI_API_KEY"):
                return []  # Skip OCR if Gemini not configured
            text = extract_dish_names_from_image(raw, mime_type="application/pdf")
            dishes = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
            return _merge_unique_preserve_order(dishes)
        except Exception:
            return []

    if is_img:
        mime = content_type.split(";")[0].strip() if content_type.startswith("image/") else "image/jpeg"
        try:
            if not os.getenv("GEMINI_API_KEY"):
                return []  # Skip OCR if Gemini not configured
            text = extract_dish_names_from_image(raw, mime_type=mime)
            dishes = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
            return _merge_unique_preserve_order(dishes)
        except Exception:
            return []

    if is_html or ("text/" in content_type):
        page_text = _extract_text_from_html_bytes(raw)
        if not page_text:
            return []
        try:
            categorized = extract_dishes_from_menu(page_text)
            flat = []
            for cat in ["appetizer", "mains", "desserts"]:
                flat.extend(categorized.get(cat, []) if isinstance(categorized, dict) else [])
            return _merge_unique_preserve_order(flat)
        except Exception:
            return []

    return []

def _dish_recommendations_for_restaurant(menu_items: List[str], user_taste_vec: List[float], diet_type: Optional[str], top_n: int = 5) -> List[Dict]:
    items = _filter_dishes_by_diet(menu_items or [], diet_type)
    items = _merge_unique_preserve_order(items)
    scored = []
    semantic = os.getenv("USE_SEMANTIC_DISH_TASTE", "false").lower() in {"1", "true", "yes", "y"}
    for dish in items[:40]:
        tv = _infer_taste_from_text_hybrid(dish, semantic=semantic)
        sim = _taste_similarity(user_taste_vec, tv)
        scored.append({"name": dish, "similarity": float(sim)})
    scored.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    return scored[: max(1, int(top_n))]

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set")
        _groq_client = Groq(api_key=groq_api_key)
    return _groq_client

def _embed_text(text: str) -> List[float]:
    model = _get_embedding_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()

def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX") or "menu-buddy"
        if not api_key or not index_name:
            raise HTTPException(status_code=500, detail="Missing Pinecone configuration")
        pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        try:
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing_indexes:
                print(f"[INFO] Creating Pinecone index: {index_name}")
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=index_name,
                    dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"[INFO] Pinecone index '{index_name}' created successfully")
        except Exception as e:
            print(f"[WARNING] Could not check/create Pinecone index: {e}")
        
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index

def _load_ingredient_flavor_map() -> Dict[str, Dict]:
    global _ingredient_flavor_map
    if _ingredient_flavor_map is not None:
        return _ingredient_flavor_map

    root = Path(__file__).resolve().parent.parent
    csv_path = root / "ingredient-flavor.csv"
    mapping: Dict[str, Dict] = {}
    if not csv_path.exists():
        _ingredient_flavor_map = mapping
        return _ingredient_flavor_map

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = (row.get("ingredient") or "").strip()
            if not ing:
                continue
            spicy = float(row.get("spicy") or 0)
            sweet = float(row.get("sweet") or 0)
            umami = float(row.get("umami") or 0)
            sour = float(row.get("sour") or 0)
            salty = float(row.get("salty") or 0)
            bitter = 0.0
            taste_vector = [sweet, salty, sour, bitter, umami, spicy]
            mapping[ing.lower()] = {
                "ingredient": ing,
                "flavor_profile": {
                    "spicy": spicy,
                    "sweet": sweet,
                    "umami": umami,
                    "sour": sour,
                    "salty": salty
                },
                "taste_vector": taste_vector
            }

    _ingredient_flavor_map = mapping
    return _ingredient_flavor_map

def _infer_taste_from_text(text: str) -> List[float]:
    m = _load_ingredient_flavor_map()
    if not text:
        return [0.0] * 6
    t = text.lower()

    hits = 0
    acc = np.zeros(6, dtype=float)
    for ing_lc, info in m.items():
        if ing_lc and re.search(r"\b" + re.escape(ing_lc) + r"\b", t):
            vec = info.get("taste_vector") or [0.0] * 6
            if len(vec) == 6:
                acc += np.array(vec, dtype=float)
                hits += 1

    if hits == 0:
        return [0.0] * 6
    out = (acc / float(hits)).tolist()
    return [float(x) for x in out]

def _infer_taste_from_text_semantic(text: str) -> List[float]:
    if not text:
        return [0.0] * 6

    cache_key = text.strip().lower()
    if cache_key in _taste_infer_cache:
        return _taste_infer_cache[cache_key]

    try:
        idx = _get_pinecone_index()
    except Exception:
        _taste_infer_cache[cache_key] = [0.0] * 6
        return _taste_infer_cache[cache_key]

    try:
        qvec = _embed_text(text)
        res = idx.query(vector=qvec, top_k=5, include_metadata=True, namespace="ingredients")
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    except Exception:
        _taste_infer_cache[cache_key] = [0.0] * 6
        return _taste_infer_cache[cache_key]

    num = np.zeros(6, dtype=float)
    den = 0.0
    for m in matches or []:
        meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
        score = float(m.get("score", 0.0)) if isinstance(m, dict) else float(getattr(m, "score", 0.0))
        tv = meta.get("taste_vector") if isinstance(meta, dict) else None
        if not isinstance(tv, list) or len(tv) != 6:
            continue
        w = max(0.0, score)
        num += w * np.array(tv, dtype=float)
        den += w

    if den <= 0.0:
        out = [0.0] * 6
    else:
        out = (num / den).tolist()

    _taste_infer_cache[cache_key] = [float(x) for x in out]
    return _taste_infer_cache[cache_key]

def _infer_taste_from_text_hybrid(text: str, semantic: bool = False) -> List[float]:
    lexical = _infer_taste_from_text(text)
    if not semantic:
        return lexical
    semantic_vec = _infer_taste_from_text_semantic(text)
    return _combine_taste_vectors(lexical, semantic_vec, secondary_weight=0.6)

def _combine_taste_vectors(primary: List[float], secondary: List[float], secondary_weight: float = 0.35) -> List[float]:
    if not primary or len(primary) != 6:
        primary = [0.0] * 6
    if not secondary or len(secondary) != 6:
        return [float(x) for x in primary]
    w = float(secondary_weight)
    w = max(0.0, min(1.0, w))
    p = np.array(primary, dtype=float)
    s = np.array(secondary, dtype=float)
    out = ((1.0 - w) * p + w * s).tolist()
    return [float(x) for x in out]

def _maybe_upsert_ingredients_to_pinecone() -> None:
    global _ingredient_upsert_done
    if _ingredient_upsert_done:
        return
    m = _load_ingredient_flavor_map()
    if not m:
        _ingredient_upsert_done = True
        return
    try:
        idx = _get_pinecone_index()
    except Exception:
        return

    vectors = []
    for ing_lc, info in m.items():
        ing = info.get("ingredient")
        if not ing:
            continue
        vec = _embed_text(ing)
        vectors.append({
            "id": f"ingredient:{ing_lc}",
            "values": vec,
            "metadata": {
                "type": "ingredient",
                "ingredient": ing,
                "flavor_profile": info.get("flavor_profile"),
                "taste_vector": info.get("taste_vector"),
            }
        })

        if len(vectors) >= 100:
            try:
                idx.upsert(vectors=vectors, namespace="ingredients")
            except Exception:
                pass
            vectors = []

    if vectors:
        try:
            idx.upsert(vectors=vectors, namespace="ingredients")
        except Exception:
            pass

    _ingredient_upsert_done = True

def _price_to_range(price: Optional[str]) -> Optional[int]:
    if not price:
        return None
    if isinstance(price, str):
        return len(price.strip()) if price.strip() else None
    return None

def _safe_lower_list(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    out = []
    for v in values:
        if v and isinstance(v, str):
            out.append(v.lower())
    return out

def _allergy_filter(menu_items: List[str], allergies: List[str]) -> bool:
    if not allergies:
        return True
    menu_text = "\n".join(menu_items).lower()
    for allergy in allergies:
        a = (allergy or "").strip().lower()
        if a and a in menu_text:
            return False
    return True

def _favorites_boost(menu_items: List[str], favorite_dishes: List[Dict]) -> float:
    if not favorite_dishes:
        return 0.0
    fav_names = [d.get("name", "") for d in favorite_dishes if isinstance(d, dict)]
    fav_names = _safe_lower_list(fav_names)
    if not fav_names:
        return 0.0
    menu_text = "\n".join(menu_items).lower()
    hits = sum(1 for f in fav_names if f and f in menu_text)
    return min(0.15, 0.03 * hits)

def _user_profile_to_taste_vector(user_profile: "UserProfile") -> List[float]:
    if not user_profile:
        return [0.0] * 6

    parts = []
    for cat in ["appetizer", "mains", "desserts"]:
        prof = getattr(user_profile, cat, None)
        if prof:
            parts.append(prof.model_dump())

    if not parts:
        return [0.0] * 6

    spicy = float(np.mean([p.get("spicy", 0.0) for p in parts]))
    sweet = float(np.mean([p.get("sweet", 0.0) for p in parts]))
    umami = float(np.mean([p.get("umami", 0.0) for p in parts]))
    sour = float(np.mean([p.get("sour", 0.0) for p in parts]))
    salty = float(np.mean([p.get("salty", 0.0) for p in parts]))
    bitter = 0.0
    return [sweet, salty, sour, bitter, umami, spicy]

def _taste_similarity(user_vec: List[float], item_vec: List[float]) -> float:
    if not user_vec or not item_vec or len(user_vec) != len(item_vec):
        return 0.0
    a = np.array(user_vec, dtype=float)
    b = np.array(item_vec, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(cosine_similarity([a], [b])[0][0])

# CORS middleware
# Allow localhost for development and any frontend URL from environment
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

# Add production frontend URL from environment if set
FRONTEND_URL = os.getenv("FRONTEND_URL", "")
if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL)

# Allow all origins in development, or specific origins in production
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
if CORS_ORIGINS != "*":
    # Parse comma-separated list of origins
    allowed_origins.extend([origin.strip() for origin in CORS_ORIGINS.split(",")])
else:
    # In development, allow all origins
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load recipes data
recipes_df = None

def load_recipes():
    global recipes_df
    if recipes_df is None:
        import os
        # Try current directory first, then parent directory
        csv_path = 'recipes_with_flavour_profiles.csv'
        if not os.path.exists(csv_path):
            csv_path = '../recipes_with_flavour_profiles.csv'
        recipes_df = pd.read_csv(csv_path)
        # Parse flavor_profile strings to dictionaries
        recipes_df['flavor_profile'] = recipes_df['flavor_profile'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return recipes_df

# Request models
class DishInput(BaseModel):
    name: str
    category: str  # "appetizer", "mains", "desserts"

class UserDishes(BaseModel):
    dishes: List[DishInput]

class MenuText(BaseModel):
    text: str
# pcsk_2366jE_NqDJXeBsSDasu5GicZQctPqRXHM7RNFjmzK5EoMXG2KZVZuyErJdNB5eSDYQktb
# Response models
class FlavorProfile(BaseModel):
    spicy: float
    sweet: float
    umami: float
    sour: float
    salty: float

class UserProfile(BaseModel):
    appetizer: FlavorProfile
    mains: FlavorProfile
    desserts: FlavorProfile
    allergies: List[str] = []
    favorite_dishes: List[DishInput] = []
    diet_type: Optional[str] = "mix"

class PreferencePrompt(BaseModel):
    prompt: str

class DummyUserUpdate(BaseModel):
    location: Optional[str] = None
    allergies: Optional[List[str]] = None
    favorite_dishes: Optional[List[DishInput]] = None
    diet_type: Optional[str] = None
    flavor_profile: Optional[UserProfile] = None

@app.get("/api/dummy-user")
def get_dummy_user(user_key: str = "default"):
    return _get_dummy_user(user_key)

@app.put("/api/dummy-user")
def update_dummy_user(data: DummyUserUpdate, user_key: str = "default"):
    u = _get_dummy_user(user_key)
    if data.location is not None:
        u["location"] = data.location
    if data.allergies is not None:
        u["allergies"] = data.allergies
    if data.favorite_dishes is not None:
        u["favorite_dishes"] = _normalize_favorite_dishes([d.model_dump() for d in data.favorite_dishes])
    if data.diet_type is not None:
        u["diet_type"] = data.diet_type
    if data.flavor_profile is not None:
        u["flavor_profile"] = data.flavor_profile.model_dump()
    return u

class RecommendationsRequest(BaseModel):
    user_profile: UserProfile
    menu_dishes: List[str]
    categorized_dishes: Optional[Dict[str, List[str]]] = None

# # Auth models
class RecipeRecommendation(BaseModel):
    id: int
    name: str
    ingredients: List[str]
    flavor_profile: FlavorProfile
    similarity_score: float
    category: str

class RecommendationsResponse(BaseModel):
    recommendations: Dict[str, List[RecipeRecommendation]]

def find_recipe_by_name(name: str, df: pd.DataFrame) -> Optional[Dict]:
    """Find a recipe by name (fuzzy matching)"""
    name_lower = name.lower().strip()
    
    # Exact match
    exact_match = df[df['name'].str.lower() == name_lower]
    if not exact_match.empty:
        return exact_match.iloc[0].to_dict()
    
    # Partial match
    partial_match = df[df['name'].str.lower().str.contains(name_lower, na=False)]
    if not partial_match.empty:
        return partial_match.iloc[0].to_dict()
    
    # Word-based matching
    name_words = set(name_lower.split())
    best_match = None
    best_score = 0
    
    for idx, row in df.iterrows():
        recipe_name = str(row['name']).lower()
        recipe_words = set(recipe_name.split())
        common_words = name_words.intersection(recipe_words)
        if common_words:
            score = len(common_words) / max(len(name_words), len(recipe_words))
            if score > best_score:
                best_score = score
                best_match = row.to_dict()
    
    return best_match if best_score > 0.3 else None

def calculate_average_flavor_profile(recipes: List[Dict]) -> Dict[str, float]:
    """Calculate average flavor profile from a list of recipes"""
    if not recipes:
        return {"spicy": 0.0, "sweet": 0.0, "umami": 0.0, "sour": 0.0, "salty": 0.0}
    
    profiles = []
    for recipe in recipes:
        if isinstance(recipe['flavor_profile'], str):
            profile = ast.literal_eval(recipe['flavor_profile'])
        else:
            profile = recipe['flavor_profile']
        profiles.append(profile)
    
    avg_profile = {
        "spicy": np.mean([p.get("spicy", 0) for p in profiles]),
        "sweet": np.mean([p.get("sweet", 0) for p in profiles]),
        "umami": np.mean([p.get("umami", 0) for p in profiles]),
        "sour": np.mean([p.get("sour", 0) for p in profiles]),
        "salty": np.mean([p.get("salty", 0) for p in profiles])
    }
    
    return {k: round(v, 2) for k, v in avg_profile.items()}

def normalize_dish_name(name: str) -> str:
    """Normalize dish name: clean, standardize capitalization, remove extra spaces"""
    if not name:
        return ""
    
    # Remove leading/trailing whitespace
    name = name.strip()
    
    # Remove extra spaces (multiple spaces to single space)
    name = re.sub(r'\s+', ' ', name)
    
    # Remove common text artifacts
    name = re.sub(r'[^\w\s\-\'&,()]', '', name)  # Keep only alphanumeric, spaces, hyphens, apostrophes, commas, parentheses, ampersands
    name = re.sub(r'\s+', ' ', name)  # Clean up spaces again
    
    # Smart capitalization: Title case but preserve certain words
    words = name.split()
    if not words:
        return ""
    
    # Words that should remain lowercase (unless first word)
    lowercase_words = {'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from'}
    
    # Words that should remain uppercase
    uppercase_words = {'USA', 'UK', 'NYC', 'BBQ', 'AI', 'CEO'}
    
    normalized_words = []
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in uppercase_words:
            normalized_words.append(word.upper())
        elif i == 0 or word_lower not in lowercase_words:
            # Title case: first letter uppercase, rest lowercase
            normalized_words.append(word.capitalize())
        else:
            normalized_words.append(word_lower)
    
    name = ' '.join(normalized_words)
    
    # Final cleanup
    name = name.strip()
    
    return name

def is_price_line(line: str) -> bool:
    """Check if a line is likely a price or price-related text"""
    line_lower = line.lower().strip()
    
    # Patterns that indicate prices
    price_patterns = [
        r'^\$?\d+\.?\d*\s*$',  # Just a number/price
        r'^\d+\.?\d*\s*\$',     # Number followed by dollar sign
        r'^\$?\d+\.?\d*\s*-\s*\$?\d+\.?\d*',  # Price range
        r'^\d+\.?\d*\s*(usd|eur|gbp|rs|rupees?)',  # Currency symbols
        r'^price',  # Starts with "price"
        r'^\d+\.?\d*\s*each',  # "X.XX each"
        r'^\d+\.?\d*\s*per',   # "X.XX per"
    ]
    
    for pattern in price_patterns:
        if re.match(pattern, line_lower):
            return True
    
    # Check if line is mostly numbers and currency symbols
    if len(line) > 0:
        non_price_chars = re.sub(r'[\d\s\$\.\,\-]', '', line)
        if len(non_price_chars) / len(line) < 0.3:  # Less than 30% non-price characters
            return True
    
    return False

_dish_validation_cache: Dict[str, bool] = {}

def _validate_dishes_with_groq(items: List[str]) -> List[str]:
    """Use Groq to intelligently filter out non-dish items (code, navigation, etc.) from extracted menu text."""
    if not items:
        return []
    
    # Check cache first
    uncached_items = []
    cached_results = []
    
    for item in items:
        cache_key = item.lower().strip()
        if cache_key in _dish_validation_cache:
            if _dish_validation_cache[cache_key]:
                cached_results.append(item)
        else:
            uncached_items.append(item)
    
    if not uncached_items:
        return cached_results
    
    # Batch validate with Groq (max 50 at a time)
    valid_dishes = []
    batch_size = 50
    
    for i in range(0, len(uncached_items), batch_size):
        batch = uncached_items[i:i+batch_size]
        
        try:
            groq_client = _get_groq_client()
            items_text = "\n".join([f"{idx+1}. {item}" for idx, item in enumerate(batch)])
            
            prompt = f"""Filter out non-food items from this list. Return ONLY the numbers of items that are actual food/dish names.

Rules:
- INCLUDE: Real food dishes, meals, appetizers, desserts, beverages that are food items
- EXCLUDE: JavaScript code, CSS, HTML tags, navigation text, buttons, headers, footers, URLs, variable names, functions, analytics code, metadata

List:
{items_text}

Respond with ONLY comma-separated numbers of valid food items (e.g., "1,3,5,7"). If none are valid, respond with "none"."""
            
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=200
            )
            
            response = completion.choices[0].message.content.strip().lower()
            
            if response == "none":
                # Mark all as invalid
                for item in batch:
                    _dish_validation_cache[item.lower().strip()] = False
            else:
                # Parse valid indices
                try:
                    valid_indices = set(int(x.strip()) - 1 for x in response.split(',') if x.strip().isdigit())
                    for idx, item in enumerate(batch):
                        is_valid = idx in valid_indices
                        _dish_validation_cache[item.lower().strip()] = is_valid
                        if is_valid:
                            valid_dishes.append(item)
                except Exception:
                    # If parsing fails, be conservative and include all
                    for item in batch:
                        _dish_validation_cache[item.lower().strip()] = True
                        valid_dishes.append(item)
        
        except Exception as e:
            print(f"[WARNING] Groq dish validation failed: {e}")
            # Fallback: use basic filtering
            for item in batch:
                item_lower = item.lower()
                # Basic filtering as fallback
                if (len(item) >= 3 and len(item) <= 80 and 
                    not any(x in item_lower for x in ['function(', '=>', 'window.', 'document.', '.push(', 'gtag', '__']) and
                    re.search(r'[a-zA-Z]', item)):
                    _dish_validation_cache[item.lower().strip()] = True
                    valid_dishes.append(item)
                else:
                    _dish_validation_cache[item.lower().strip()] = False
    
    return cached_results + valid_dishes

def is_dish_name(line: str) -> bool:
    """Quick basic validation before Groq batch processing"""
    line_clean = line.strip()
    
    # Basic checks
    if len(line_clean) < 2 or len(line_clean) > 150:
        return False
    
    if is_price_line(line_clean):
        return False
    
    if not re.search(r'[a-zA-Z]', line_clean):
        return False
    
    return True

def extract_dishes_from_menu(menu_text: str) -> Dict[str, List[str]]:
    """Extract dish names from menu text and categorize them with Groq-powered filtering"""
    lines = menu_text.split('\n')
    
    # Category synonyms mapping
    appetizer_synonyms = ['appetizer', 'appetiser', 'appetizers', 'appetisers', 
                         'starter', 'starters', 'small plates', 'small plate',
                         'tapas', 'hors d\'oeuvres', 'hors d\'oeuvre', 'hors d oeuvres',
                         'beginning', 'beginnings', 'first course', 'first courses']
    mains_synonyms = ['main', 'mains', 'main course', 'main courses', 
                     'entree', 'entrees', 'big plates', 'big plate',
                     'large plates', 'large plate', 'mains course', 'main dish',
                     'main dishes', 'second course', 'second courses']
    dessert_synonyms = ['dessert', 'desserts', 'sweet', 'sweets', 
                       'pudding', 'puddings', 'finale', 'finales',
                       'sweet course', 'sweet courses', 'after dinner']
    
    categorized_dishes = {
        "appetizer": [],
        "mains": [],
        "desserts": []
    }
    
    current_category = None
    
    for line in lines:
        line_original = line.strip()
        if not line_original:
            continue
        
        line_lower = line_original.lower()
        
        # Check if this line is a category header
        is_appetizer_header = any(syn in line_lower for syn in appetizer_synonyms)
        is_mains_header = any(syn in line_lower for syn in mains_synonyms)
        is_dessert_header = any(syn in line_lower for syn in dessert_synonyms)
        
        if is_appetizer_header:
            current_category = "appetizer"
            continue
        elif is_mains_header:
            current_category = "mains"
            continue
        elif is_dessert_header:
            current_category = "desserts"
            continue
        
        # Skip if not a valid dish name
        if not is_dish_name(line_original):
            continue
        
        # Clean and extract dish name (remove prices, numbers, etc.)
        # Remove numbering at start (1., 2), etc.)
        line_clean = re.sub(r'^\d+[\.\)]\s*', '', line_original)
        
        # Remove prices - more comprehensive patterns
        # Remove $XX.XX at end
        line_clean = re.sub(r'\s*\$?\d+\.?\d*\s*$', '', line_clean)
        # Remove - $XX.XX
        line_clean = re.sub(r'\s*-\s*\$?\d+\.?\d*\s*$', '', line_clean)
        # Remove (XX.XX) or [XX.XX]
        line_clean = re.sub(r'\s*[\(\[].*?\d+\.?\d*.*?[\)\]]\s*$', '', line_clean)
        # Remove standalone price patterns
        line_clean = re.sub(r'\s+\$\d+\.?\d*\s*', ' ', line_clean)
        
        # Remove common suffixes that might be prices
        line_clean = re.sub(r'\s+\d+\.?\d*\s*(usd|eur|gbp|rs|rupees?|each|per)\s*$', '', line_clean, flags=re.IGNORECASE)
        
        # Clean up
        line_clean = line_clean.strip()
        
        # Skip if too short after cleaning
        if len(line_clean) < 2:
            continue
        
        # Collect for batch validation
        normalized_name = normalize_dish_name(line_clean)
        
        if not normalized_name or len(normalized_name) < 2:
            continue
        
        if is_price_line(normalized_name):
            continue
        
        # Store with category info for later
        if current_category:
            if normalized_name not in categorized_dishes[current_category]:
                categorized_dishes[current_category].append(normalized_name)
        else:
            # Try to infer category from dish name
            dish_lower = normalized_name.lower()
            dessert_keywords = ['cake', 'pie', 'ice cream', 'pudding', 'chocolate', 'cookie', 
                              'brownie', 'tart', 'mousse', 'custard', 'flan', 'sorbet',
                              'cheesecake', 'tiramisu', 'gelato', 'sundae', 'parfait',
                              'creme brulee', 'creme brûlée', 'baklava', 'cannoli']
            appetizer_keywords = ['salad', 'soup', 'dip', 'bruschetta', 'samosa', 'spring roll',
                                'wings', 'nachos', 'quesadilla', 'hummus', 'guacamole',
                                'appetizer', 'starter', 'tapas', 'antipasto', 'mezze',
                                'crostini', 'canape', 'canapé']
            
            if any(keyword in dish_lower for keyword in dessert_keywords):
                if dish_lower not in categorized_dishes["desserts"]:
                    categorized_dishes["desserts"].append(normalized_name)
            elif any(keyword in dish_lower for keyword in appetizer_keywords):
                if dish_lower not in categorized_dishes["appetizer"]:
                    categorized_dishes["appetizer"].append(normalized_name)
            else:
                if dish_lower not in categorized_dishes["mains"]:
                    categorized_dishes["mains"].append(normalized_name)
    
    # Batch validate all collected dishes with Groq
    all_dishes = categorized_dishes["appetizer"] + categorized_dishes["mains"] + categorized_dishes["desserts"]
    valid_dishes = _validate_dishes_with_groq(all_dishes)
    valid_set = set(valid_dishes)
    
    # Filter each category to only include validated dishes
    categorized_dishes["appetizer"] = [d for d in categorized_dishes["appetizer"] if d in valid_set]
    categorized_dishes["mains"] = [d for d in categorized_dishes["mains"] if d in valid_set]
    categorized_dishes["desserts"] = [d for d in categorized_dishes["desserts"] if d in valid_set]
    
    # Limit each category to 20 dishes and remove duplicates
    for category in categorized_dishes:
        # Remove duplicates while preserving order
        seen = set()
        unique_dishes = []
        for dish in categorized_dishes[category]:
            dish_lower = dish.lower()
            if dish_lower not in seen:
                seen.add(dish_lower)
                unique_dishes.append(dish)
        categorized_dishes[category] = unique_dishes[:20]
    
    return categorized_dishes

def calculate_similarity(profile1: Dict, profile2: Dict) -> float:
    """Calculate cosine similarity between two flavor profiles"""
    keys = ['spicy', 'sweet', 'umami', 'sour', 'salty']
    vec1 = np.array([profile1.get(k, 0) for k in keys])
    vec2 = np.array([profile2.get(k, 0) for k in keys])
    
    # Avoid division by zero
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    
    return float(cosine_similarity([vec1], [vec2])[0][0])

@app.get("/")
def read_root():
    return {"message": "Swaad Recipe Recommendation API"}

def get_gemini_client():
    """Get or initialize Gemini client"""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        # Client automatically uses GEMINI_API_KEY from environment
        return genai.Client()
    except Exception as e:
        print(f"Warning: Gemini client initialization failed: {e}")
        return None

def extract_dish_names_from_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Extract ONLY dish names from menu image using Gemini API with structured JSON output"""
    
    if not GEMINI_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Gemini API not available. Please install google-genai: pip install google-genai"
        )
    
    gemini_client = get_gemini_client()
    if not gemini_client:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key not configured. Please set GEMINI_API_KEY environment variable or add it to .env file. Get your API key from https://aistudio.google.com"
        )
    
    try:
        # Use JSON response mode for structured output
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                """You are a menu extraction expert. Extract ONLY the dish/item names from this menu image.

CRITICAL RULES - FOLLOW THESE EXACTLY:
1. Extract ONLY the dish/item names - nothing else
2. DO NOT include:
   - Prices (e.g., $9.85, $14.00, $12.50)
   - Descriptions (e.g., "Eggs your style", "served with", ingredient lists)
   - Allergen symbols (e.g., Ⓦ, Ⓓ, Ⓔ, Ⓕ, Ⓢ, Ⓝ, Ⓩ, Ⓥ, Ⓦ)
   - Section headers (e.g., "Breakfast", "Tartines", "Shakshuka", "BRUNCH")
   - Restaurant name, hours, footer text, or any other metadata
   - Partial words or fragments
3. DO include:
   - Complete dish names exactly as they appear (e.g., "Breakfast Sandwich", "French Toast")
   - Variants if clearly separate dishes (e.g., "French Toast - Sweet", "French Toast - Savory")
   - Full dish names even if they span multiple words

EXAMPLES OF CORRECT EXTRACTION:
✅ "Breakfast Sandwich" (correct)
✅ "Croissant Breakfast Sandwich" (correct)
✅ "Halloumi Sunny-Side Breakfast Sandwich" (correct)
❌ "$9.85" (WRONG - this is a price)
❌ "Eggs your style" (WRONG - this is a description)
❌ "Breakfast" (WRONG - this is a section header)
❌ "BRUNCH" (WRONG - this is a section header)
❌ "Served all day" (WRONG - this is metadata)

Return a JSON object with this exact format:
{
  "dishes": [
    "Dish Name 1",
    "Dish Name 2",
    "Dish Name 3"
  ]
}

Return ONLY valid JSON, no other text or explanation."""
            ],
            config=config
        )
        
        if response and response.text:
            # Parse JSON response
            data = json.loads(response.text)
            
            # Extract dish names from JSON
            if isinstance(data, dict) and "dishes" in data:
                dishes = data["dishes"]
            elif isinstance(data, list):
                dishes = data
            else:
                # Fallback: try to extract from text if JSON parsing fails
                dishes = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            # Filter out empty strings and return as newline-separated string
            dish_names = [dish.strip() for dish in dishes if dish and dish.strip()]
            
            if not dish_names:
                raise HTTPException(
                    status_code=400,
                    detail="No dish names found in menu image. Please ensure the image contains a readable menu."
                )
            
            return '\n'.join(dish_names)
        else:
            raise HTTPException(
                status_code=500,
                detail="Gemini API returned empty response"
            )
            
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract from plain text response
        try:
            text = response.text if response and response.text else ""
            
            # Try to find JSON in the text (sometimes Gemini adds extra text)
            import re
            json_match = re.search(r'\{[^{}]*"dishes"[^{}]*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if isinstance(data, dict) and "dishes" in data:
                        dishes = data["dishes"]
                        dish_names = [dish.strip() for dish in dishes if dish and dish.strip()]
                        if dish_names:
                            return '\n'.join(dish_names)
                except:
                    pass
            
            # Fallback: try to extract dish names from plain text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            dish_names = []
            skip_keywords = ['$', 'price', 'contains', 'allergen', 'served', 'hours', 'brunch', 'breakfast', 'tartines', 'shakshuka', 'menu', 'restaurant']
            
            for line in lines:
                line_lower = line.lower()
                # Skip lines that are clearly not dish names
                if any(skip in line_lower for skip in skip_keywords):
                    continue
                # Skip lines that are mostly numbers or symbols
                if re.match(r'^[\d\s\$\.\,\-]+$', line):
                    continue
                # Skip very short or very long lines
                if len(line) < 3 or len(line) > 80:
                    continue
                # Skip lines that look like prices
                if re.match(r'^\$?\d+\.?\d*\s*$', line):
                    continue
                
                dish_names.append(line)
            
            if dish_names:
                return '\n'.join(dish_names)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to extract dish names. JSON parsing error: {str(e)}. Please ensure GEMINI_API_KEY is set correctly."
                )
        except HTTPException:
            raise
        except Exception as fallback_error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract dish names: {str(fallback_error)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting dish names from image: {str(e)}"
        )

@app.post("/api/process-menu")
def process_menu_text(menu: MenuText):
    """Extract dish names from menu text and categorize them"""
    categorized_dishes = extract_dishes_from_menu(menu.text)
    # Flatten for backward compatibility, but also return categorized
    all_dishes = categorized_dishes["appetizer"] + categorized_dishes["mains"] + categorized_dishes["desserts"]
    return {
        "dishes": all_dishes,
        "categorized": categorized_dishes
    }

@app.post("/api/upload-menu-image")
async def upload_menu_image(file: UploadFile = File(...)):
    """Upload menu image and extract ONLY dish names using Gemini API"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image bytes
    try:
        image_bytes = await file.read()
        
        # Determine MIME type
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            # Try to infer from filename
            filename = file.filename.lower() if file.filename else ""
            if filename.endswith('.png'):
                mime_type = "image/png"
            elif filename.endswith('.webp'):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Default
        
        # Extract dish names from image using Gemini API
        extracted_text = extract_dish_names_from_image(image_bytes, mime_type)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract text from image. Please ensure the image is clear and contains readable text.")
        
        # Process the extracted text
        categorized_dishes = extract_dishes_from_menu(extracted_text)
        all_dishes = categorized_dishes["appetizer"] + categorized_dishes["mains"] + categorized_dishes["desserts"]
        
        return {
            "extracted_text": extracted_text,
            "dishes": all_dishes,
            "categorized": categorized_dishes
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/api/recommendations", response_model=RecommendationsResponse)
def get_recommendations(request: RecommendationsRequest):
    """Get recommendations based on user profile and menu dishes"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    client = Groq(api_key=groq_api_key)

    user_profile = request.user_profile
    menu_dishes = request.menu_dishes
    categorized_dishes = request.categorized_dishes

    if categorized_dishes:
        dish_categories = categorized_dishes
    else:
        dish_categories = {
            "appetizer": [],
            "mains": [],
            "desserts": []
        }
        for dish in menu_dishes:
            dish_lower = dish.lower()
            dessert_keywords = ['cake', 'pie', 'ice cream', 'pudding', 'chocolate', 'cookie',
                              'brownie', 'tart', 'mousse', 'custard', 'flan', 'sorbet']
            appetizer_keywords = ['salad', 'soup', 'dip', 'bruschetta', 'samosa', 'spring roll',
                                'wings', 'nachos', 'quesadilla', 'hummus', 'guacamole']

            if any(kw in dish_lower for kw in dessert_keywords):
                dish_categories["desserts"].append(dish)
            elif any(kw in dish_lower for kw in appetizer_keywords):
                dish_categories["appetizer"].append(dish)
            else:
                dish_categories["mains"].append(dish)

    dish_entries = []
    seen = set()
    for category, dishes in dish_categories.items():
        for dish_name in dishes:
            key = (category, (dish_name or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            dish_entries.append({"name": dish_name, "category": category})

    system_prompt = """
    You are a culinary expert.
    For each dish name provided, return a flavor profile (0.0 to 1.0) and a short ingredient list.
    
    Output JSON format:
    {
      "dishes": [
        {
          "name": "string",
          "category": "appetizer/mains/desserts",
          "ingredients": ["string"],
          "flavor_profile": {"spicy": 0.0-1.0, "sweet": 0.0-1.0, "umami": 0.0-1.0, "sour": 0.0-1.0, "salty": 0.0-1.0}
        }
      ]
    }
    """

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({"dishes": dish_entries})}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        parsed = json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dish profiles: {str(e)}")

    dish_profiles = {}
    for d in parsed.get("dishes", []) if isinstance(parsed, dict) else []:
        name = (d.get("name") or "").strip()
        category = (d.get("category") or "mains").strip()
        prof = d.get("flavor_profile") or {}
        ingredients = d.get("ingredients") or []
        dish_profiles[(category, name.lower())] = {
            "flavor_profile": {
                "spicy": float(prof.get("spicy", 0.5)),
                "sweet": float(prof.get("sweet", 0.5)),
                "umami": float(prof.get("umami", 0.5)),
                "sour": float(prof.get("sour", 0.5)),
                "salty": float(prof.get("salty", 0.5)),
            },
            "ingredients": [str(x) for x in ingredients if x][:10]
        }

    recommendations = {
        "appetizer": [],
        "mains": [],
        "desserts": []
    }

    next_id = 1
    for category in ["appetizer", "mains", "desserts"]:
        user_profile_dict = user_profile.model_dump().get(category, {})
        scored_items = []
        for dish_name in dish_categories.get(category, []):
            key = (category, (dish_name or "").strip().lower())
            info = dish_profiles.get(key)
            if not info:
                continue

            if hasattr(user_profile, "allergies") and user_profile.allergies:
                if not _allergy_filter(info.get("ingredients", []), user_profile.allergies):
                    continue

            similarity = calculate_similarity(user_profile_dict, info["flavor_profile"])
            scored_items.append((dish_name, info, similarity))

        scored_items.sort(key=lambda x: x[2], reverse=True)
        for dish_name, info, score in scored_items[:5]:
            recommendations[category].append({
                "id": next_id,
                "name": dish_name,
                "ingredients": info.get("ingredients", []),
                "flavor_profile": info.get("flavor_profile", {}),
                "similarity_score": round(float(score), 3),
                "category": category
            })
            next_id += 1

    return {"recommendations": recommendations}

@app.get("/api/search-recipes")
def search_recipes(query: str, limit: int = 10):
    """Search for recipes by name"""
    df = load_recipes()
    query_lower = query.lower()
    
    matches = df[df['name'].str.lower().str.contains(query_lower, na=False)]
    
    results = []
    for idx, row in matches.head(limit).iterrows():
        if isinstance(row['ingredients'], str):
            ingredients = ast.literal_eval(row['ingredients'])
        else:
            ingredients = row['ingredients']
        
        if isinstance(row['flavor_profile'], str):
            flavor_profile = ast.literal_eval(row['flavor_profile'])
        else:
            flavor_profile = row['flavor_profile']
        
        results.append({
            "id": int(row['id']),
            "name": row['name'],
            "ingredients": ingredients,
            "flavor_profile": flavor_profile
        })
    
    return {"recipes": results}

class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    favorite_dishes: Optional[List[DishInput]] = None
    location: Optional[str] = None
    max_results: Optional[int] = None
    diet_type: Optional[str] = None
    user_key: Optional[str] = "default"

@app.post("/api/chat")
async def chat_with_yelp(request: ChatRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    groq_client = _get_groq_client()

    _maybe_upsert_ingredients_to_pinecone()

    # Dummy user map is the source of truth.
    # If the frontend sends metadata, we treat it as a sync into the dummy map.
    _sync_dummy_user_from_request(request)
    user_key = (request.user_key or "default").strip() or "default"
    dummy_user = _get_dummy_user(user_key)
    dummy_profile = _dummy_user_to_user_profile(dummy_user)

    allergies = (dummy_user.get("allergies") or []) if isinstance(dummy_user, dict) else []
    favorite_dishes = (dummy_user.get("favorite_dishes") or []) if isinstance(dummy_user, dict) else []
    diet_type = (dummy_user.get("diet_type") if isinstance(dummy_user, dict) else None) or "mix"

    print(f"[DEBUG] Using dummy user: {user_key}")
    print(f"[DEBUG] Allergies: {allergies}")
    print(f"[DEBUG] Favorite dishes: {favorite_dishes}")
    print(f"[DEBUG] Diet type: {diet_type}")

    if dummy_profile:
        allergies = dummy_profile.allergies or allergies
        favorite_dishes = [d.model_dump() for d in (dummy_profile.favorite_dishes or [])] or favorite_dishes
        diet_type = dummy_profile.diet_type or diet_type

    user_taste_vec = _user_profile_to_taste_vector(dummy_profile) if dummy_profile else [0.0] * 6
    fav_text = ""
    if favorite_dishes:
        try:
            fav_text = " ".join([(d.name if hasattr(d, "name") else (d.get("name") if isinstance(d, dict) else str(d))) for d in favorite_dishes])
        except Exception:
            fav_text = ""
    semantic_user = os.getenv("USE_SEMANTIC_INGREDIENT_TASTE", "true").lower() in {"1", "true", "yes", "y"}
    inferred_user = _infer_taste_from_text_hybrid(fav_text, semantic=semantic_user)
    user_taste_vec = _combine_taste_vectors(user_taste_vec, inferred_user, secondary_weight=0.35)

    is_first_turn = not request.chat_id
    pending_query = dummy_user.get("pending_query") if isinstance(dummy_user, dict) else None
    if is_first_turn and pending_query and request.query and not request.location:
        # User likely replied with location-only; reuse stored query.
        request.location = request.query
        request.query = pending_query
        dummy_user["location"] = request.location
        dummy_user["pending_query"] = None
    user_db_location = (dummy_user.get("location") if isinstance(dummy_user, dict) else None) or ""
    fallback_location = request.location
    if not fallback_location and user_db_location:
        fallback_location = user_db_location
    
    print(f"[DEBUG] user_key: {user_key}")
    print(f"[DEBUG] user_db_location: {user_db_location}")
    print(f"[DEBUG] request.location: {request.location}")
    print(f"[DEBUG] fallback_location: {fallback_location}")

    # Set final_max_results
    final_max_results = request.max_results or 10
    print(f"[DEBUG] max_results set to: {final_max_results}")

    # Check if location is provided
    if is_first_turn and not fallback_location:
        if isinstance(dummy_user, dict):
            dummy_user["pending_query"] = request.query
        return {
            "response": {
                "text": "Please share your location (city or ZIP/postal code) so I can find restaurants near you."
            },
            "chat_id": request.chat_id
        }

    # Query Pinecone directly for restaurant recommendations
    print(f"[DEBUG] Querying Pinecone for restaurant recommendations")

    # Initialize response structure
    ai_json = {
        "response": {
            "text": f"Here are some great restaurant recommendations for you in {fallback_location}!"
        },
        "chat_id": request.chat_id
    }

    ranked = []
    try:
        pc_index = _get_pinecone_index()
        qvec = _embed_text(request.query)
        top_k = max(final_max_results, 10)
        print(f"[DEBUG] Querying Pinecone with top_k={top_k}")
        query_res = pc_index.query(vector=qvec, top_k=top_k, include_metadata=True)
        matches = query_res.get("matches", []) if isinstance(query_res, dict) else getattr(query_res, "matches", [])
        print(f"[DEBUG] Pinecone returned {len(matches)} matches")

        for m in matches:
            meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
            score = float(m.get("score", 0.0)) if isinstance(m, dict) else float(getattr(m, "score", 0.0))
            menu_items = meta.get("menu_items") or []
            menu_items = _filter_dishes_by_diet(menu_items, diet_type)
            if not menu_items:
                continue
            if not _allergy_filter(menu_items, allergies):
                continue
            location = meta.get("location")
            if location is None:
                loc_json = meta.get("location_json")
                if isinstance(loc_json, str) and loc_json:
                    try:
                        location = json.loads(loc_json)
                    except Exception:
                        location = loc_json
            coordinates = meta.get("coordinates")
            if coordinates is None:
                coords_json = meta.get("coordinates_json")
                if isinstance(coords_json, str) and coords_json:
                    try:
                        coordinates = json.loads(coords_json)
                    except Exception:
                        coordinates = coords_json
            taste_vec = None
            try:
                t0 = meta.get("taste_0")
                t1 = meta.get("taste_1")
                t2 = meta.get("taste_2")
                t3 = meta.get("taste_3")
                t4 = meta.get("taste_4")
                t5 = meta.get("taste_5")
                if all(isinstance(x, (int, float)) for x in [t0, t1, t2, t3, t4, t5]):
                    taste_vec = [float(t0), float(t1), float(t2), float(t3), float(t4), float(t5)]
            except Exception:
                taste_vec = None
            if not taste_vec:
                taste_vec = [0.0] * 6
            tscore = _taste_similarity(user_taste_vec, taste_vec)
            boost = _favorites_boost(menu_items, [d.model_dump() if hasattr(d, "model_dump") else d for d in favorite_dishes] if favorite_dishes else [])
            combined = score + 0.35 * tscore + boost
            ranked.append({
                "id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None),
                "name": meta.get("name"),
                "url": meta.get("url"),
                "avg_rating": meta.get("avg_rating"),
                "price_range": meta.get("price_range"),
                "cuisine_types": meta.get("cuisine_types"),
                "location": location,
                "coordinates": coordinates,
                "menu_items": menu_items,
                "popular_dishes": meta.get("popular_dishes"),
                "taste_vector": taste_vec,
                "recommended_dishes": _dish_recommendations_for_restaurant(menu_items, user_taste_vec, diet_type, top_n=5),
                "photos": meta.get("photos"),
                "menu_url": meta.get("menu_url"),
                "score": combined
            })

        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        print(f"[DEBUG] Total ranked restaurants: {len(ranked)}")
        # Limit to requested max_results
        ranked = ranked[:final_max_results]
        print(f"[DEBUG] Returning top {len(ranked)} recommendations")
    except Exception as e:
        print(f"[DEBUG] Error in ranking: {e}")
        import traceback
        traceback.print_exc()
        ranked = []

    ai_json["menu_buddy"] = {
        "recommendations": ranked
    }
    
    # Post-process Yelp AI response text to match filtered results
    try:
        original_text = ai_json.get("response", {}).get("text", "")
        if original_text and ranked:
            restaurant_names = [r.get("name") for r in ranked if r.get("name")]
            diet_label = "vegetarian" if diet_type in {"veg", "vegetarian"} else diet_type or "any diet"
            
            rewrite_prompt = f"""Rewrite this restaurant recommendation text to:
1. Match the actual {len(ranked)} restaurants shown: {', '.join(restaurant_names[:5])}
2. Only mention dishes suitable for {diet_label} diet (NO meat, fish, eggs, or animal products if vegetarian)
3. Keep the tone friendly and helpful
4. Be concise (2-3 sentences max)

Original text: {original_text}

Rewritten text:"""
            
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": rewrite_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=200
            )
            
            rewritten_text = completion.choices[0].message.content.strip()
            ai_json["response"]["text"] = rewritten_text
            print(f"[DEBUG] Rewrote response text for {diet_label} diet")
    except Exception as e:
        print(f"[DEBUG] Failed to rewrite response text: {e}")

    return ai_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

