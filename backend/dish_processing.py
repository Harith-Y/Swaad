"""
Dish extraction, classification, and filtering utilities.
"""
from typing import List, Optional, Dict, Tuple
import re
import json
from groq import Groq
from config import GROQ_API_KEY


# Global Groq client
_groq_client = None

# Cache for dish diet classification
_dish_diet_cache: Dict[str, str] = {}

# Cache for dish validation
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


def detect_diet_from_query(query: str) -> Optional[str]:
    """
    Detect diet preference from query string using Groq.
    Returns 'veg', 'non-veg', or None.
    """
    # Fast path for obvious keywords
    query_lower = query.lower()
    if any(x in query_lower for x in ["non-veg", "non veg", "nonvegetarian", "non vegetarian", "meat", "chicken", "beef", "pork"]):
        return "non-veg"
    if any(x in query_lower for x in ["veg", "vegetarian", "pure veg", "vegan"]):
        return "veg"

    # Use Groq for complex queries
    try:
        client = get_groq_client()
        prompt = f"""Analyze the dietary preference in this query.
Query: "{query}"

Rules:
1. If the user explicitly asks for vegetarian/vegan food (e.g. "no meat", "plant based"), return "veg".
2. If the user explicitly asks for meat/non-veg food (e.g. "meat lover", "carnivore"), return "non-veg".
3. If no specific diet is mentioned, return "none".

Response (only "veg", "non-veg", or "none"):"""

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=10
        )

        result = completion.choices[0].message.content.strip().lower()
        if result in ["veg", "non-veg"]:
            return result

        return None
    except Exception as e:
        print(f"[ERROR] Diet detection failed: {e}")
        return None


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


def filter_dishes_by_allergy(dishes: List[str], allergies: List[str]) -> List[str]:
    """
    Filter dishes that are safe for the given allergies using Groq.
    """
    if not dishes or not allergies:
        return dishes

    try:
        client = get_groq_client()
        safe_dishes = []
        batch_size = 50

        for i in range(0, len(dishes), batch_size):
            batch = dishes[i:i+batch_size]

            dishes_text = "\n".join([f"{idx+1}. {d}" for idx, d in enumerate(batch)])
            allergies_text = ", ".join(allergies)

            prompt = f"""Identify which of these dishes are SAFE for someone with these allergies: {allergies_text}.

Rules:
1. Analyze the likely ingredients of each dish.
2. If a dish likely contains an allergen (e.g. "Pesto" contains nuts/dairy, "Carbonara" contains egg/dairy/pork), exclude it.
3. Be strict. Safety first.
4. Return ONLY the numbers of the SAFE dishes.
5. Return comma-separated numbers (e.g. "1,3,5").
6. If none are safe, return "none".

List:
{dishes_text}

Response:"""

            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=100
            )

            result = completion.choices[0].message.content.strip().lower()

            if result == "none":
                continue

            try:
                indices = [int(x.strip()) - 1 for x in result.split(',') if x.strip().isdigit()]
                for idx in indices:
                    if 0 <= idx < len(batch):
                        safe_dishes.append(batch[idx])
            except Exception:
                # Fallback
                for d in batch:
                    if allergy_filter(d, allergies):
                        safe_dishes.append(d)

        return safe_dishes

    except Exception as e:
        print(f"[ERROR] Groq allergy filter failed: {e}")
        # Fallback
        return [d for d in dishes if allergy_filter(d, allergies)]


def classify_dish_diet_with_groq(dish_name: str) -> str:
    """
    Classify a dish as 'veg' or 'non-veg' using Groq LLM.
    Returns: 'veg' or 'non-veg'
    Uses caching to avoid repeated API calls.
    """
    if not dish_name or not isinstance(dish_name, str):
        return "veg"  # Default to veg if invalid

    cache_key = dish_name.lower().strip()
    if cache_key in _dish_diet_cache:
        return _dish_diet_cache[cache_key]

    try:
        groq_client = get_groq_client()
        prompt = f"""Classify this dish as either 'veg' or 'non-veg'.

Dish: {dish_name}

Rules:
- 'non-veg' includes: meat, poultry, fish, seafood, eggs, and any animal products (except dairy)
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

        _dish_diet_cache[cache_key] = classification
        print(f"[DEBUG] Classified '{dish_name}' as '{classification}'")
        return classification

    except Exception as e:
        print(f"[WARNING] Groq diet classification failed for '{dish_name}': {e}")
        # Fallback to basic keyword check
        t = dish_name.lower()
        nonveg_keywords = {"chicken", "beef", "pork", "bacon", "ham", "turkey", "lamb",
                          "mutton", "duck", "fish", "salmon", "tuna", "shrimp", "prawn",
                          "crab", "lobster", "egg", "meat", "seafood"}
        classification = "non-veg" if any(k in t for k in nonveg_keywords) else "veg"
        _dish_diet_cache[cache_key] = classification
        return classification


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


def extract_dish_from_query(query: str) -> Optional[str]:
    """
    Use regex and Groq to extract the specific dish name from a user query.
    Returns None if no specific dish is requested.
    """
    query_lower = query.lower().strip()

    # 1. Fast Regex Extraction for common patterns
    patterns = [
        r"i want to eat (?:a |an )?(.+)",
        r"i want (?:a |an )?(.+)",
        r"craving (?:for )?(?:a |an )?(.+)",
        r"looking for (?:a |an )?(.+)",
        r"where can i (?:get|find|eat) (?:a |an )?(.+)",
        r"show me (?:places with |restaurants with )?(.+)",
        r"do you have (?:a |an )?(.+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            candidate = match.group(1).strip()
            # Clean up common trailing words
            candidate = re.sub(r"\s+(?:near|in|at|from)\s+.*$", "", candidate).strip()
            # Remove punctuation
            candidate = re.sub(r"[^\w\s]", "", candidate).strip()

            # Filter out generic terms that regex might catch
            generic_terms = {"food", "something", "anything", "restaurant", "place", "restaurants", "places", "dinner", "lunch", "breakfast"}
            if candidate and candidate not in generic_terms and len(candidate) > 2 and len(candidate) < 50:
                return candidate

    # 2. Fallback to LLM for complex queries
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

        # Sanity check: reject long repetitive strings (hallucinations)
        if len(result) > 50 or len(set(result.split())) < len(result.split()) / 2:
             return None

        if len(result) < 2:
            return None

        return result

    except Exception as e:
        print(f"[ERROR] Dish extraction failed: {e}")
        return None



def extract_location_from_query(query: str) -> Optional[str]:
    """
    Extract location from the query text.
    Returns: Location string if found, None otherwise
    """
    query_lower = query.lower()

    # Location patterns
    patterns = [
        r'\bnear\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
        r'\bin\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
        r'\bat\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
        r'\baround\s+(.+?)(?:\s*,|\s+also|\s+and|\s*$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            location = match.group(1).strip()
            # Filter out common non-location words
            non_locations = ["me", "here", "there", "my place", "home"]
            if location not in non_locations and len(location) > 2:
                return location

    return None


def classify_intent(query: str) -> str:
    """
    Classify user intent using Groq.
    Returns: 'dish_search', 'restaurant_search', 'greeting', 'other'
    """
    try:
        client = get_groq_client()

        prompt = f"""Classify the user's intent from this query into ONE category.

Query: "{query}"

Categories:
- dish_search: User is looking for a specific dish (e.g., "I want pizza", "where can I get sushi")
- restaurant_search: User is looking for a restaurant by name (e.g., "show me Olive Garden", "is there a McDonald's nearby")
- greeting: User is greeting or asking general questions (e.g., "hello", "how are you", "what can you do")
- other: Anything else

Return ONLY the category name, nothing else.

Response:"""

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=10
        )

        result = completion.choices[0].message.content.strip().lower()

        valid_intents = ["dish_search", "restaurant_search", "greeting", "other"]
        for intent in valid_intents:
            if intent in result:
                return intent

        return "other"

    except Exception as e:
        print(f"[ERROR] Intent classification failed: {e}")
        return "other"


def is_relevant_query(query: str) -> bool:
    """
    Check if query is relevant to food/restaurant search using Groq AI.
    Returns True if query is about food/restaurants, False otherwise.
    """
    query_lower = query.lower()

    # Fast path: obvious food/restaurant keywords
    relevant_keywords = [
        "food", "eat", "restaurant", "dish", "meal", "hungry", "craving",
        "lunch", "dinner", "breakfast", "cuisine", "menu", "order",
        "pizza", "burger", "sushi", "pasta", "chicken", "veg", "non-veg",
        "taste", "flavor", "spicy", "sweet", "savory"
    ]

    # Fast path: obvious non-food keywords
    irrelevant_keywords = [
        "tourist", "sightseeing", "museum", "park", "beach", "hotel",
        "shopping", "mall", "attraction", "landmark", "monument"
    ]

    # If contains irrelevant keywords and no relevant keywords, reject immediately
    has_irrelevant = any(keyword in query_lower for keyword in irrelevant_keywords)
    has_relevant = any(keyword in query_lower for keyword in relevant_keywords)

    if has_irrelevant and not has_relevant:
        return False

    if has_relevant:
        return True

    # Use Groq for ambiguous queries
    try:
        client = get_groq_client()
        prompt = f"""Is this query about food, restaurants, or dining?
Query: "{query}"

Rules:
1. Return "yes" if asking about food, restaurants, dishes, meals, or dining.
2. Return "no" if asking about tourist attractions, hotels, shopping, or general travel.
3. Return "no" if asking about non-food places to visit.

Response (only "yes" or "no"):"""

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=5
        )

        result = completion.choices[0].message.content.strip().lower()
        return result == "yes"

    except Exception as e:
        print(f"[ERROR] Relevance check failed: {e}")
        # Default to True to avoid blocking valid queries
        return True


def check_location_match(user_location: str, restaurant_location: str) -> bool:
    """
    Check if user location matches restaurant location using Groq.
    """
    if not user_location or not restaurant_location:
        return False

    # Ensure inputs are strings
    if not isinstance(user_location, str): user_location = str(user_location)
    if not isinstance(restaurant_location, str): restaurant_location = str(restaurant_location)

    # Normalize for comparison
    u_loc = user_location.lower().strip()
    r_loc = restaurant_location.lower().strip()
    
    # Fast path 1: Direct substring match (improved)
    # Extract key location terms (city names, not just first word)
    # Common patterns: "New York", "San Francisco", "Los Angeles"
    
    # Check for exact city match (case-insensitive)
    # Split by comma to get city part
    u_city = u_loc.split(',')[0].strip()
    r_city = r_loc.split(',')[0].strip()
    
    # Check if user's city appears in restaurant's location
    # Use word boundaries to avoid "San" matching "San Antonio" when user wants "San Francisco"
    if u_city and u_city in r_loc:
        # Verify it's a full city name match, not just a prefix
        # Check if followed by comma, space at end, or other word boundary
        import re
        pattern = r'\b' + re.escape(u_city) + r'\b'
        if re.search(pattern, r_loc):
            print(f"[DEBUG] Fast path match: '{u_city}' found in '{r_loc}'")
            return True
    
    # Fast path 2: Check for country/region mismatches
    # Check for US vs non-US locations
    us_indicators = [', us', 'united states', ', ny', ', ca', ', tx', ', fl']
    non_us_countries = ['mexico', 'canada', 'uk', 'france', 'italy', 'spain', 'india', 'china', 'japan']
    
    u_has_us = any(ind in r_loc for ind in us_indicators)
    u_has_non_us = any(country in u_loc for country in non_us_countries)
    
    r_has_us = any(ind in r_loc for ind in us_indicators)
    r_has_non_us = any(country in r_loc for country in non_us_countries)
    
    # If one is clearly US and the other is clearly non-US, reject
    if (u_has_non_us and r_has_us) or (u_has_us and r_has_non_us):
        print(f"[DEBUG] Fast path rejection: country mismatch (US vs non-US)")
        return False
    
    # Fast path 3: Regional grouping for nearby cities
    # Some cities are close enough to be considered the same region
    city_regions = {
        # Bay Area
        'san francisco': 'bay_area',
        'san jose': 'bay_area',
        'oakland': 'bay_area',
        'berkeley': 'bay_area',
        'palo alto': 'bay_area',
        'mountain view': 'bay_area',
        'sunnyvale': 'bay_area',
        # NYC Metro
        'new york': 'nyc_metro',
        'brooklyn': 'nyc_metro',
        'manhattan': 'nyc_metro',
        'queens': 'nyc_metro',
        'bronx': 'nyc_metro',
        'staten island': 'nyc_metro',
        # Other major cities (no regional grouping)
        'los angeles': 'la',
        'chicago': 'chicago',
        'boston': 'boston',
        'seattle': 'seattle',
        'portland': 'portland',
        'austin': 'austin',
        'miami': 'miami',
        'atlanta': 'atlanta'
    }
    
    user_region = None
    restaurant_region = None
    
    for city, region in city_regions.items():
        if city in u_loc:
            user_region = region
        if city in r_loc:
            restaurant_region = region
    
    # If both regions are identified
    if user_region and restaurant_region:
        # Same region = match
        if user_region == restaurant_region:
            print(f"[DEBUG] Fast path match: same region ({user_region})")
            return True
        # Different regions = no match
        else:
            print(f"[DEBUG] Fast path rejection: {user_region} != {restaurant_region}")
            return False

    # Fast path 4: State code mismatch (e.g., CA vs NY)
    # Extract state codes (2 letters after comma)
    import re
    user_state = re.search(r',\s*([A-Z]{2})\b', user_location)
    rest_state = re.search(r',\s*([A-Z]{2})\b', restaurant_location)
    
    if user_state and rest_state:
        if user_state.group(1) != rest_state.group(1):
            print(f"[DEBUG] Fast path rejection: state mismatch {user_state.group(1)} != {rest_state.group(1)}")
            return False
    
    print(f"[DEBUG] Fast path inconclusive, calling Groq for location match")
    try:
        client = get_groq_client()
        prompt = f"""Determine if these two locations refer to the same area or if one is inside the other.
Location A: "{user_location}"
Location B: "{restaurant_location}"

Rules:
1. Return "yes" if they match (e.g. "NYC" and "New York", "Manhattan" and "New York City").
2. Return "yes" if one is inside the other (e.g. "Brooklyn" and "New York").
3. Return "no" if they are different cities or far apart.

Response (only "yes" or "no"):"""

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=5
        )

        result = completion.choices[0].message.content.strip().lower()
        return "yes" in result
    except Exception as e:
        print(f"[ERROR] Location match failed: {e}")
        # Fallback to simple check


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


def extract_dish_and_restaurant(query: str) -> Optional[Tuple[str, str]]:
    """
    Extract dish and restaurant name from query using Groq.
    Returns (dish, restaurant) or None.
    """
    # Fast check for 'from', 'at', 'in'
    query_lower = query.lower()
    if not any(x in query_lower for x in ['from', 'at', 'in', 'restaurant']):
        return None

    try:
        client = get_groq_client()
        prompt = f"""Extract the dish name and restaurant name from this query.
Query: "{query}"

Rules:
1. Return ONLY a JSON object with keys "dish" and "restaurant".
2. If either is missing, return "none" for that key.
3. Example: "Pizza from Dominos" -> {{"dish": "pizza", "restaurant": "dominos"}}
4. Example: "Burger at Shake Shack" -> {{"dish": "burger", "restaurant": "shake shack"}}
5. Example: "I want pasta" -> {{"dish": "pasta", "restaurant": "none"}}

Response:"""

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(completion.choices[0].message.content)
        dish = result.get("dish", "none")
        restaurant = result.get("restaurant", "none")

        if dish != "none" and restaurant != "none":
            return (dish, restaurant)

        return None

    except Exception as e:
        print(f"[ERROR] Dish/Restaurant extraction failed: {e}")
        return None

