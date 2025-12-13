"""
Dish extraction, classification, and filtering utilities.
"""
from typing import List, Optional, Dict
import re
import json
from groq import Groq
from config import GROQ_API_KEY


# Global Groq client
_groq_client = None
_dish_validation_cache: Dict[str, bool] = {}


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


def extract_dish_from_query(query: str) -> Optional[str]:
    """
    Use Groq to extract the specific dish name from a user query.
    Returns None if no specific dish is requested.
    """
    try:
        client = get_groq_client()
        
        prompt = f"""Extract the specific food dish the user is asking for from this query.
Query: "{query}"

Rules:
1. If the user is asking for a specific dish (e.g., "I want pizza", "where can I get sushi"), return ONLY the dish name (e.g., "pizza", "sushi").
2. If the user is asking for a cuisine (e.g., "Italian food"), return "cuisine:italian".
3. If the user is just greeting or asking general questions, return "none".
4. Remove words like "to eat", "I want", "looking for", "best", "delicious".
5. Return ONLY the extracted term in lowercase.

Example 1: "I want to eat a burger" -> "burger"
Example 2: "Show me places with pasta" -> "pasta"
Example 3: "I'm hungry" -> "none"
Example 4: "Best italian restaurants" -> "cuisine:italian"

Response:"""

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=20
        )
        
        result = completion.choices[0].message.content.strip().lower()
        
        if result == "none" or result.startswith("cuisine:"):
            return None
            
        # Clean up any quotes or extra whitespace
        result = result.replace('"', '').replace("'", "").strip()
        
        if len(result) < 2:
            return None
            
        return result
            
    except Exception as e:
        print(f"[ERROR] Dish extraction failed: {e}")
        return None


def is_relevant_query(query: str) -> bool:
    """
    Check if the query is relevant to the Swaad AI use case (food, restaurants, etc.).
    """
    # Quick check for very short queries or greetings to save API calls
    query_lower = query.lower().strip()
    if len(query_lower) < 2:
        return True
        
    greetings = {"hi", "hello", "hey", "hola", "greetings", "yo", "sup", "thanks", "thank you", "bye", "goodbye"}
    if query_lower in greetings or any(query_lower.startswith(g + " ") for g in greetings):
        return True

    try:
        client = get_groq_client()
        
        prompt = f"""
        You are a classifier for a food recommendation AI.
        Determine if the following user query is relevant to:
        1. Food, dishes, ingredients, or cuisines.
        2. Restaurants, dining out, or ordering food.
        3. Cooking, recipes, or dietary preferences (vegan, allergies, etc.).
        4. Greetings (hi, hello) or closing (bye, thanks).
        5. General conversation that could lead to food recommendations (e.g. "I'm hungry", "what should I eat").

        Query: "{query}"

        Respond with ONLY "yes" if it is relevant, or "no" if it is off-topic (e.g. coding, politics, math, weather, sports, general knowledge not about food).
        """

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=5
        )
        
        result = completion.choices[0].message.content.strip().lower()
        return "yes" in result

    except Exception as e:
        print(f"[ERROR] Relevance check failed: {e}")
        # Fail open (assume relevant) if AI fails to avoid blocking legitimate queries
        return True


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


def validate_dishes_with_groq(items: List[str]) -> List[str]:
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
            groq_client = get_groq_client()
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


async def extract_dishes_from_menu(menu_text: str) -> Dict[str, List[str]]:
    """Extract dish names from menu text and categorize them using Groq LLM."""
    if not menu_text:
        return {"appetizer": [], "mains": [], "desserts": []}

    # Limit text length to avoid token limits (approx 15k chars ~ 3-4k tokens)
    # We'll take the first chunk which usually contains the menu
    truncated_text = menu_text[:15000] 

    groq_client = get_groq_client()
    if not groq_client:
        print("[DEBUG] Groq client not available for menu extraction")
        return {"appetizer": [], "mains": [], "desserts": []}

    prompt = f"""
    You are an AI menu parser. Extract food dishes from the provided menu text and categorize them into 'appetizer', 'mains', and 'desserts'.
    
    Rules:
    1. Return ONLY a valid JSON object.
    2. Keys must be "appetizer", "mains", "desserts".
    3. Values must be lists of strings (dish names).
    4. Clean the dish names (remove prices, ingredients, descriptions).
    5. Ignore beverages, sides, and non-food text.
    6. If a dish fits multiple categories, pick the best one.
    
    Menu Text:
    {truncated_text}
    """

    try:
        print("[DEBUG] Sending menu text to Groq for extraction...")
        
        # Note: This is a synchronous call in the original code, but wrapped in async function
        # We'll keep it synchronous for now as Groq client is sync
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        response_text = completion.choices[0].message.content
        data = json.loads(response_text)
        
        # Validate structure
        result = {
            "appetizer": data.get("appetizer", []),
            "mains": data.get("mains", []),
            "desserts": data.get("desserts", [])
        }
        
        # Additional validation
        for cat in result:
            if not isinstance(result[cat], list):
                result[cat] = []
            # Filter out empty strings
            result[cat] = [d for d in result[cat] if isinstance(d, str) and d.strip()]
            
        return result

    except Exception as e:
        print(f"[ERROR] Menu extraction failed: {e}")
        return {"appetizer": [], "mains": [], "desserts": []}
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

