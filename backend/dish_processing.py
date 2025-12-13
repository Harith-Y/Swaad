"""
Dish extraction, classification, and filtering utilities.
"""
from typing import List, Optional
import re
from groq import Groq
from config import GROQ_API_KEY


# Global Groq client
_groq_client = None


def get_groq_client():
    """Get or initialize Groq client."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# Non-vegetarian keywords for filtering
NON_VEG_KEYWORDS = [
    "chicken", "mutton", "lamb", "beef", "pork", "fish", "prawn", "shrimp",
    "crab", "lobster", "meat", "bacon", "sausage", "ham", "turkey", "duck",
    "egg", "eggs", "omelette", "omelet", "seafood", "salmon", "tuna"
]


def is_nonveg_text(text: str) -> bool:
    """Check if text contains non-vegetarian keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in NON_VEG_KEYWORDS)


def filter_dishes_by_diet(dishes: List[str], diet_type: Optional[str]) -> List[str]:
    """Filter dishes based on dietary preferences."""
    if not dishes:
        return []
    
    if not diet_type or diet_type in {"mix", "any", "all"}:
        return dishes
    
    if diet_type in {"veg", "vegetarian"}:
        return [d for d in dishes if not is_nonveg_text(d)]
    
    if diet_type in {"non-veg", "nonveg", "non_veg"}:
        return [d for d in dishes if is_nonveg_text(d)]
    
    return dishes


def allergy_filter(menu_items: List[str], allergies: List[str]) -> bool:
    """
    Check if menu items are safe for user allergies.
    Returns True if safe (no allergens found), False if allergens detected.
    """
    if not allergies:
        return True
    
    menu_text = " ".join(menu_items).lower()
    for allergen in allergies:
        if allergen.lower() in menu_text:
            return False
    
    return True


def classify_dish_with_groq(dish_name: str) -> str:
    """Classify dish into category using Groq AI."""
    try:
        client = get_groq_client()
        
        prompt = f"""Classify this dish into ONE category: appetizer, mains, or desserts.
Dish: {dish_name}

Respond with ONLY the category name (appetizer, mains, or desserts), nothing else."""

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=10
        )
        
        category = completion.choices[0].message.content.strip().lower()
        
        if category in {"appetizer", "appetizers"}:
            return "appetizer"
        elif category in {"mains", "main", "main course", "entree"}:
            return "mains"
        elif category in {"desserts", "dessert"}:
            return "desserts"
        else:
            return "mains"  # Default
            
    except Exception as e:
        print(f"[ERROR] Groq classification failed: {e}")
        return "mains"


def normalize_dish_name(name: str) -> str:
    """Normalize dish name for comparison."""
    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[^\w\s-]', '', name)
    return name


def is_price_line(line: str) -> bool:
    """Check if line is likely a price indicator."""
    price_patterns = [
        r'\$\d+',
        r'₹\d+',
        r'\d+\.\d{2}',
        r'price:',
        r'cost:',
    ]
    return any(re.search(pattern, line.lower()) for pattern in price_patterns)


def is_dish_name(line: str) -> bool:
    """Check if line is likely a dish name."""
    line = line.strip()
    
    if not line or len(line) < 3:
        return False
    
    if is_price_line(line):
        return False
    
    # Skip section headers
    if line.lower() in {"appetizers", "mains", "desserts", "beverages", "drinks", "menu"}:
        return False
    
    # Skip lines that are too long (likely descriptions)
    if len(line) > 100:
        return False
    
    return True


def merge_unique_preserve_order(items: List[str]) -> List[str]:
    """Merge list items while preserving order and removing duplicates."""
    seen = set()
    result = []
    for item in items:
        item_lower = item.lower().strip()
        if item_lower and item_lower not in seen:
            seen.add(item_lower)
            result.append(item)
    return result

