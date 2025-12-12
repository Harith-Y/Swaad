from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
try:
    from email_validator import validate_email
    EMAIL_VALIDATION_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATION_AVAILABLE = False
    # Use string instead if email-validator not available
    EmailStr = str
from typing import List, Dict, Optional
import pandas as pd
import ast
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sqlalchemy.orm import Session
from database import get_db, User, Base, engine, YelpRawResponse
from auth import (
    verify_password, get_password_hash, create_access_token,
    get_user_by_email, get_user_by_username, get_current_user
)
from dotenv import load_dotenv
import os
from groq import Groq
import httpx
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()

try:
    from google_auth import verify_google_token
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_AUTH_AVAILABLE = bool(GOOGLE_CLIENT_ID)
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
    GOOGLE_CLIENT_ID = None
from datetime import timedelta

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

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def _embed_text(text: str) -> List[float]:
    model = _get_embedding_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()

def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX")
        if not api_key or not index_name:
            raise HTTPException(status_code=500, detail="Missing Pinecone configuration")
        pc = Pinecone(api_key=api_key)
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index

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
    p = user_profile.mains.model_dump() if user_profile and user_profile.mains else {}
    spicy = float(p.get("spicy", 0.0))
    sweet = float(p.get("sweet", 0.0))
    umami = float(p.get("umami", 0.0))
    sour = float(p.get("sour", 0.0))
    salty = float(p.get("salty", 0.0))
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

class PreferencePrompt(BaseModel):
    prompt: str

class RecommendationsRequest(BaseModel):
    user_profile: UserProfile
    menu_dishes: List[str]
    categorized_dishes: Optional[Dict[str, List[str]]] = None

# Auth models
class UserSignup(BaseModel):
    email: EmailStr
    username: str
    password: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "johndoe",
                "password": "securepassword123"
            }
        }

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str

class GoogleLoginRequest(BaseModel):
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    google_id: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    flavor_profile: Optional[UserProfile] = None
    favorite_dishes: List[DishInput] = []

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

def is_dish_name(line: str) -> bool:
    """Check if a line is likely a dish name"""
    line_clean = line.strip()
    
    # Too short or too long
    if len(line_clean) < 2 or len(line_clean) > 80:
        return False
    
    # Skip if it's clearly a price
    if is_price_line(line_clean):
        return False
    
    # Skip if it's mostly numbers
    if re.match(r'^\d+[\.\)]\s*$', line_clean):
        return False
    
    # Skip common non-dish keywords
    skip_keywords = [
        'menu', 'drink', 'beverage', 'wine', 'beer', 'cocktail', 'coffee', 'tea', 
        'juice', 'allergen', 'contains', 'gluten', 'vegan', 'vegetarian',
        'page', 'copyright', 'tel', 'phone', 'email', 'website', 'www',
        'hours', 'open', 'closed', 'monday', 'tuesday', 'wednesday', 'thursday',
        'friday', 'saturday', 'sunday', 'am', 'pm'
    ]
    
    line_lower = line_clean.lower()
    if any(keyword in line_lower for keyword in skip_keywords):
        # But allow if it's part of a dish name (e.g., "Vegan Burger")
        if not any(keyword == line_lower for keyword in skip_keywords):
            # Check if it's a standalone keyword
            if line_lower in skip_keywords:
                return False
    
    # Should have at least one letter
    if not re.search(r'[a-zA-Z]', line_clean):
        return False
    
    # Should not be mostly special characters
    special_char_ratio = len(re.sub(r'[\w\s]', '', line_clean)) / len(line_clean) if line_clean else 0
    if special_char_ratio > 0.5:
        return False
    
    return True

def extract_dishes_from_menu(menu_text: str) -> Dict[str, List[str]]:
    """Extract dish names from menu text and categorize them with improved filtering"""
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
        
        # Normalize the dish name
        normalized_name = normalize_dish_name(line_clean)
        
        if not normalized_name or len(normalized_name) < 2:
            continue
        
        # Skip if it's still a price after cleaning
        if is_price_line(normalized_name):
            continue
        
        # If we have a current category, use it; otherwise try to infer
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
            
            if any(kw in dish_lower for kw in dessert_keywords):
                if normalized_name not in categorized_dishes["desserts"]:
                    categorized_dishes["desserts"].append(normalized_name)
            elif any(kw in dish_lower for kw in appetizer_keywords):
                if normalized_name not in categorized_dishes["appetizer"]:
                    categorized_dishes["appetizer"].append(normalized_name)
            else:
                # Default to mains if uncertain
                if normalized_name not in categorized_dishes["mains"]:
                    categorized_dishes["mains"].append(normalized_name)
    
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

# Initialize database
Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Swaad Recipe Recommendation API"}

# Authentication endpoints
@app.post("/api/auth/signup", response_model=Token)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Create a new user account"""
    try:
        # Check if email already exists
        if get_user_by_email(db, user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if username already exists
        if get_user_by_username(db, user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Validate password length
        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters"
            )
        
        # Check byte length (bcrypt limit is 72 bytes, not characters)
        password_bytes = user_data.password.encode('utf-8')
        if len(password_bytes) > 72:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is too long (maximum 72 bytes). Please use a shorter password."
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            favorite_dishes=[]
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # Create access token
        access_token_expires = timedelta(minutes=30 * 24 * 60)  # 30 days
        access_token = create_access_token(
            data={"sub": db_user.id}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": db_user.id,
            "username": db_user.username
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Signup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating account: {str(e)}"
        )

@app.post("/api/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=30 * 24 * 60)  # 30 days
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }

@app.post("/api/auth/google", response_model=Token)
def google_login(google_data: GoogleLoginRequest, db: Session = Depends(get_db)):
    """Login or signup with Google OAuth"""
    email = google_data.email
    name = google_data.name or ''
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required"
        )
    
    # Validate email format
    if '@' not in email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Check if user exists by email
    user = get_user_by_email(db, email)
    
    if not user:
        # Create new user with Google account
        # Generate a username from email or name
        username_base = name.lower().replace(' ', '_').replace('.', '_') if name else email.split('@')[0]
        # Remove special characters
        username_base = ''.join(c for c in username_base if c.isalnum() or c == '_')
        username = username_base[:30]  # Limit username length
        counter = 1
        
        # Ensure username is unique
        while get_user_by_username(db, username):
            username = f"{username_base[:25]}{counter}"
            counter += 1
            if counter > 1000:  # Safety limit
                username = f"user_{email.split('@')[0][:20]}"
                break
        
        # Create user without password (Google-authenticated users don't need password)
        db_user = User(
            email=email,
            username=username,
            hashed_password="",  # Empty for Google-authenticated users
            favorite_dishes=[]
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        user = db_user
    else:
        # Existing user - update username if it's empty
        if not user.username:
            username_base = name.lower().replace(' ', '_') if name else email.split('@')[0]
            username_base = ''.join(c for c in username_base if c.isalnum() or c == '_')
            user.username = username_base[:30]
            db.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=30 * 24 * 60)  # 30 days
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }

@app.get("/api/auth/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    flavor_profile = None
    if current_user.flavor_profile:
        flavor_profile = UserProfile(**current_user.flavor_profile)
    
    favorite_dishes = []
    if current_user.favorite_dishes:
        favorite_dishes = [DishInput(**dish) for dish in current_user.favorite_dishes]
    
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "flavor_profile": flavor_profile,
        "favorite_dishes": favorite_dishes
    }

# User profile management endpoints
@app.post("/api/user/profile", response_model=UserProfile)
def save_user_profile(
    user_dishes: UserDishes,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save or update user's flavor profile"""
    df = load_recipes()
    
    categories = {
        "appetizer": [],
        "mains": [],
        "desserts": []
    }
    
    for dish in user_dishes.dishes:
        recipe = find_recipe_by_name(dish.name, df)
        if recipe:
            categories[dish.category].append(recipe)
    
    # Calculate average profiles for each category
    appetizer_profile = calculate_average_flavor_profile(categories["appetizer"])
    mains_profile = calculate_average_flavor_profile(categories["mains"])
    desserts_profile = calculate_average_flavor_profile(categories["desserts"])
    
    user_profile = {
        "appetizer": appetizer_profile,
        "mains": mains_profile,
        "desserts": desserts_profile
    }
    
    # Save to database
    current_user.flavor_profile = user_profile
    current_user.favorite_dishes = [dish.dict() for dish in user_dishes.dishes]
    db.commit()
    
    return user_profile

@app.get("/api/user/profile", response_model=UserProfile)
def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get user's saved flavor profile"""
    if not current_user.flavor_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No flavor profile found. Please create one first."
        )
    return UserProfile(**current_user.flavor_profile)

@app.put("/api/user/profile/dishes")
def update_user_dishes(
    user_dishes: UserDishes,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's favorite dishes and recalculate profile"""
    # Save dishes
    current_user.favorite_dishes = [dish.dict() for dish in user_dishes.dishes]
    
    # Recalculate profile
    df = load_recipes()
    categories = {
        "appetizer": [],
        "mains": [],
        "desserts": []
    }
    
    for dish in user_dishes.dishes:
        recipe = find_recipe_by_name(dish.name, df)
        if recipe:
            categories[dish.category].append(recipe)
    
    appetizer_profile = calculate_average_flavor_profile(categories["appetizer"])
    mains_profile = calculate_average_flavor_profile(categories["mains"])
    desserts_profile = calculate_average_flavor_profile(categories["desserts"])
    
    current_user.flavor_profile = {
        "appetizer": appetizer_profile,
        "mains": mains_profile,
        "desserts": desserts_profile
    }
    
    db.commit()
    
    return {
        "message": "Profile updated successfully",
        "flavor_profile": current_user.flavor_profile,
        "favorite_dishes": current_user.favorite_dishes
    }

@app.delete("/api/user/profile/dishes")
def remove_dish_from_profile(
    dish_name: str,
    category: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove a dish from user's profile"""
    if not current_user.favorite_dishes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No dishes found in profile"
        )
    
    # Remove dish
    updated_dishes = [
        dish for dish in current_user.favorite_dishes
        if not (dish.get("name") == dish_name and dish.get("category") == category)
    ]
    
    if len(updated_dishes) == len(current_user.favorite_dishes):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dish not found in profile"
        )
    
    current_user.favorite_dishes = updated_dishes
    
    # Recalculate profile
    if updated_dishes:
        df = load_recipes()
        categories = {
            "appetizer": [],
            "mains": [],
            "desserts": []
        }
        
        for dish in updated_dishes:
            recipe = find_recipe_by_name(dish["name"], df)
            if recipe:
                categories[dish["category"]].append(recipe)
        
        appetizer_profile = calculate_average_flavor_profile(categories["appetizer"])
        mains_profile = calculate_average_flavor_profile(categories["mains"])
        desserts_profile = calculate_average_flavor_profile(categories["desserts"])
        
        current_user.flavor_profile = {
            "appetizer": appetizer_profile,
            "mains": mains_profile,
            "desserts": desserts_profile
        }
    else:
        current_user.flavor_profile = None
    
    db.commit()
    
    return {
        "message": "Dish removed successfully",
        "flavor_profile": current_user.flavor_profile,
        "favorite_dishes": current_user.favorite_dishes
    }

@app.post("/api/create-profile", response_model=UserProfile)
def create_user_profile(user_dishes: UserDishes):
    """Create user flavor profile from liked dishes (works for both authenticated and guest users)"""
    df = load_recipes()
    
    categories = {
        "appetizer": [],
        "mains": [],
        "desserts": []
    }
    
    found_recipes = []
    not_found = []
    
    for dish in user_dishes.dishes:
        recipe = find_recipe_by_name(dish.name, df)
        if recipe:
            categories[dish.category].append(recipe)
            found_recipes.append(dish.name)
        else:
            not_found.append(dish.name)
    
    # Calculate average profiles for each category
    appetizer_profile = calculate_average_flavor_profile(categories["appetizer"])
    mains_profile = calculate_average_flavor_profile(categories["mains"])
    desserts_profile = calculate_average_flavor_profile(categories["desserts"])
    
    return {
        "appetizer": appetizer_profile,
        "mains": mains_profile,
        "desserts": desserts_profile
    }

@app.post("/api/create-profile-ai", response_model=UserProfile)
def create_profile_ai(data: PreferencePrompt):
    """Create user flavor profile from natural language prompt using Groq"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    client = Groq(api_key=groq_api_key)
    
    system_prompt = """
    You are a culinary expert AI. Analyze the user's taste preferences to create a flavor profile.
    
    Output JSON format:
    {
        "dishes": [{"name": "Dish Name", "category": "appetizer/mains/desserts"}],
        "allergies": ["list", "of", "allergies"],
        "flavor_profile": {
            "appetizer": {"spicy": 0.0-1.0, "sweet": 0.0-1.0, "umami": 0.0-1.0, "sour": 0.0-1.0, "salty": 0.0-1.0},
            "mains": {"spicy": 0.0-1.0, "sweet": 0.0-1.0, "umami": 0.0-1.0, "sour": 0.0-1.0, "salty": 0.0-1.0},
            "desserts": {"spicy": 0.0-1.0, "sweet": 0.0-1.0, "umami": 0.0-1.0, "sour": 0.0-1.0, "salty": 0.0-1.0}
        }
    }
    
    Instructions:
    1. Extract any dishes mentioned and categorize them.
    2. Extract any allergies mentioned.
    3. Estimate the flavor profile (0.0 to 1.0) for each category based on the user's description and mentioned dishes.
    4. If a category is not mentioned, provide a balanced profile or infer from general preferences.
    5. "spicy", "sweet", "umami", "sour", "salty" are the 5 dimensions.
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.prompt}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        
        dishes = []
        for d in result.get("dishes", []):
            dishes.append({
                "name": d.get("name", "Unknown"),
                "category": d.get("category", "mains")
            })
        
        # Validate and structure the response
        return {
            "appetizer": result.get("flavor_profile", {}).get("appetizer", {"spicy": 0.5, "sweet": 0.5, "umami": 0.5, "sour": 0.5, "salty": 0.5}),
            "mains": result.get("flavor_profile", {}).get("mains", {"spicy": 0.5, "sweet": 0.5, "umami": 0.5, "sour": 0.5, "salty": 0.5}),
            "desserts": result.get("flavor_profile", {}).get("desserts", {"spicy": 0.5, "sweet": 0.5, "umami": 0.5, "sour": 0.5, "salty": 0.5}),
            "allergies": result.get("allergies", []),
            "favorite_dishes": dishes
        }
        
    except Exception as e:
        print(f"Groq Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing preferences: {str(e)}")

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

@app.post("/api/chat")
async def chat_with_yelp(request: ChatRequest, db: Session = Depends(get_db)):
    groq_api_key = os.getenv("GROQ_API_KEY")
    yelp_api_key = os.getenv("YELP_API_KEY")
    
    if not groq_api_key or not yelp_api_key:
        raise HTTPException(status_code=500, detail="Missing API keys")

    groq_client = Groq(api_key=groq_api_key)

    system_prompt = """
    You output JSON only.
    Create two things:
    1) yelp_ai_body: the JSON body to send to https://api.yelp.com/ai/chat/v2
    2) business_search: params for https://api.yelp.com/v3/businesses/search

    business_search rules:
    - Always include "limit": 50 and "offset": 50.
    - Extract a best-guess "term" (e.g., pizza, ramen, sushi).
    - Prefer an explicit location string if present; otherwise use the provided fallback_location if present.

    Output schema:
    {
      "yelp_ai_body": {"query": "...", "chat_id": "...", "user_context": {"latitude": 0.0, "longitude": 0.0}, "request_context": {"max_results": 10}},
      "business_search": {"term": "...", "location": "...", "latitude": 0.0, "longitude": 0.0, "limit": 50, "offset": 50}
    }
    """

    user_message = {
        "query": request.query,
        "chat_id": request.chat_id,
        "fallback_location": request.location,
        "fallback_max_results": request.max_results
    }

    yelp_ai_body = {"query": request.query}
    if request.chat_id:
        yelp_ai_body["chat_id"] = request.chat_id
    business_search = {"term": None, "location": request.location, "limit": 50, "offset": 50}

    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_message)}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        parsed = json.loads(completion.choices[0].message.content)
        if isinstance(parsed, dict):
            yelp_ai_body = parsed.get("yelp_ai_body") or yelp_ai_body
            business_search = parsed.get("business_search") or business_search
    except Exception:
        pass

    if request.chat_id and "chat_id" not in yelp_ai_body:
        yelp_ai_body["chat_id"] = request.chat_id
    if request.max_results:
        yelp_ai_body.setdefault("request_context", {})
        yelp_ai_body["request_context"]["max_results"] = request.max_results
    else:
        yelp_ai_body.setdefault("request_context", {})
        yelp_ai_body["request_context"].setdefault("max_results", 10)

    search_term = business_search.get("term") if isinstance(business_search, dict) else None
    search_location = business_search.get("location") if isinstance(business_search, dict) else None
    search_lat = business_search.get("latitude") if isinstance(business_search, dict) else None
    search_lng = business_search.get("longitude") if isinstance(business_search, dict) else None
    if not search_location:
        search_location = request.location

    user_context = yelp_ai_body.get("user_context") if isinstance(yelp_ai_body, dict) else None
    uc_lat = user_context.get("latitude") if isinstance(user_context, dict) else None
    uc_lng = user_context.get("longitude") if isinstance(user_context, dict) else None
    if search_lat is None:
        search_lat = uc_lat
    if search_lng is None:
        search_lng = uc_lng

    if not search_location and (search_lat is None or search_lng is None):
        return {
            "response": {
                "text": "Please share your location (city or ZIP/postal code) so I can find restaurants near you."
            },
            "chat_id": request.chat_id
        }

    ai_headers = {"Authorization": f"Bearer {yelp_api_key}", "Content-Type": "application/json"}
    yelp_ai_url = "https://api.yelp.com/ai/chat/v2"

    try:
        async with httpx.AsyncClient() as http_client:
            ai_resp = await http_client.post(yelp_ai_url, json=yelp_ai_body, headers=ai_headers)
            ai_resp.raise_for_status()
            ai_json = ai_resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Yelp API Error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    try:
        db.add(YelpRawResponse(query=request.query, endpoint="ai/chat/v2", request_params=yelp_ai_body, response_json=ai_json))
        db.commit()
    except Exception:
        db.rollback()

    v3_search_json = None
    if search_term and (search_location or (search_lat is not None and search_lng is not None)):
        v3_url = "https://api.yelp.com/v3/businesses/search"
        params = {"term": search_term, "limit": 50, "offset": 50}
        if search_location:
            params["location"] = search_location
        else:
            params["latitude"] = search_lat
            params["longitude"] = search_lng
        try:
            async with httpx.AsyncClient() as http_client:
                v3_resp = await http_client.get(v3_url, headers={"Authorization": f"Bearer {yelp_api_key}"}, params=params)
                v3_resp.raise_for_status()
                v3_search_json = v3_resp.json()
        except Exception:
            v3_search_json = None

        if v3_search_json is not None:
            try:
                db.add(YelpRawResponse(query=request.query, endpoint="v3/businesses/search", request_params=params, response_json=v3_search_json))
                db.commit()
            except Exception:
                db.rollback()

    restaurants = []
    businesses = []
    try:
        entities = ai_json.get("entities") or []
        if entities and isinstance(entities, list) and isinstance(entities[0], dict):
            businesses = entities[0].get("businesses") or []
    except Exception:
        businesses = []

    if not businesses and v3_search_json and isinstance(v3_search_json, dict):
        businesses = v3_search_json.get("businesses") or []

    try:
        ingest_count = int((yelp_ai_body.get("request_context") or {}).get("max_results") or 10)
    except Exception:
        ingest_count = 10
    ingest_count = max(1, min(50, ingest_count))
    businesses = businesses[:ingest_count]

    if businesses:
        biz_payload = []
        for b in businesses:
            if not isinstance(b, dict):
                continue
            categories = b.get("categories") or []
            cuisine_types = []
            for c in categories:
                if isinstance(c, dict) and c.get("title"):
                    cuisine_types.append(c.get("title"))
            attrs = b.get("attributes") if isinstance(b.get("attributes"), dict) else {}
            menu_url = attrs.get("MenuUrl") if attrs else None
            summaries = b.get("summaries") if isinstance(b.get("summaries"), dict) else {}
            contextual = b.get("contextual_info") if isinstance(b.get("contextual_info"), dict) else {}
            review_snip = contextual.get("review_snippet")
            photos = contextual.get("photos") if isinstance(contextual.get("photos"), list) else []
            photo_urls = [p.get("original_url") for p in photos if isinstance(p, dict) and p.get("original_url")]

            biz_payload.append({
                "id": b.get("id"),
                "name": b.get("name"),
                "cuisine_types": cuisine_types,
                "price": b.get("price"),
                "rating": b.get("rating"),
                "url": b.get("url"),
                "location": b.get("location"),
                "coordinates": b.get("coordinates"),
                "menu_url": menu_url,
                "summaries": summaries,
                "review_snippet": review_snip,
                "photo_urls": photo_urls
            })

        taste_prompt = """
        You are a food recommender.
        For each restaurant, infer:
        - menu_items: 10-20 plausible menu items for that restaurant
        - popular_dishes: 5-10 popular dishes
        - taste_vector: 6 floats in [0,1] ordered as [sweet, salty, sour, bitter, umami, spicy]

        Output JSON:
        {"restaurants": [{"id": "...", "menu_items": ["..."], "popular_dishes": ["..."], "taste_vector": [0,0,0,0,0,0]}]}
        """

        enrich_map = {}
        try:
            completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": taste_prompt},
                    {"role": "user", "content": json.dumps({"restaurants": biz_payload})}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            enriched = json.loads(completion.choices[0].message.content)
            for r in enriched.get("restaurants", []) if isinstance(enriched, dict) else []:
                rid = r.get("id")
                if rid:
                    enrich_map[str(rid)] = r
        except Exception:
            enrich_map = {}

        try:
            pc_index = _get_pinecone_index()
        except Exception:
            pc_index = None

        for b in biz_payload:
            rid = str(b.get("id") or "")
            if not rid:
                continue

            extra = enrich_map.get(rid, {})
            menu_items = extra.get("menu_items") if isinstance(extra.get("menu_items"), list) else []
            popular_dishes = extra.get("popular_dishes") if isinstance(extra.get("popular_dishes"), list) else []
            taste_vector = extra.get("taste_vector") if isinstance(extra.get("taste_vector"), list) else [0.0] * 6

            embed_text = "\n".join([
                str(b.get("name") or ""),
                " ".join([str(x) for x in (b.get("cuisine_types") or [])]),
                str((b.get("summaries") or {}).get("long") or ""),
                "\n".join([str(x) for x in menu_items[:30]]),
                "\n".join([str(x) for x in popular_dishes[:20]]),
            ])
            vec = _embed_text(embed_text)

            restaurant_obj = {
                "id": rid,
                "name": b.get("name"),
                "location": b.get("location"),
                "coordinates": b.get("coordinates"),
                "cuisine_types": b.get("cuisine_types") or [],
                "price_range": _price_to_range(b.get("price")),
                "avg_rating": b.get("rating"),
                "url": b.get("url"),
                "menu_items": menu_items,
                "popular_dishes": popular_dishes,
                "taste_vector": taste_vector,
                "photos": b.get("photo_urls") or [],
                "menu_url": b.get("menu_url")
            }
            restaurants.append(restaurant_obj)

            if pc_index is not None:
                metadata = {
                    "name": restaurant_obj["name"],
                    "url": restaurant_obj["url"],
                    "avg_rating": restaurant_obj["avg_rating"],
                    "price_range": restaurant_obj["price_range"],
                    "cuisine_types": restaurant_obj["cuisine_types"],
                    "location": restaurant_obj["location"],
                    "coordinates": restaurant_obj["coordinates"],
                    "menu_items": restaurant_obj["menu_items"],
                    "popular_dishes": restaurant_obj["popular_dishes"],
                    "taste_vector": restaurant_obj["taste_vector"],
                    "photos": restaurant_obj["photos"],
                    "menu_url": restaurant_obj["menu_url"],
                }
                try:
                    pc_index.upsert(vectors=[{"id": rid, "values": vec, "metadata": metadata}])
                except Exception:
                    pass

    ranked = []
    try:
        pc_index = _get_pinecone_index()
        qvec = _embed_text(request.query)
        top_k = 20
        query_res = pc_index.query(vector=qvec, top_k=top_k, include_metadata=True)
        matches = query_res.get("matches", []) if isinstance(query_res, dict) else getattr(query_res, "matches", [])

        allergies = []
        favorite_dishes = []
        if request.user_profile:
            allergies = request.user_profile.allergies or []
            favorite_dishes = request.user_profile.favorite_dishes or []
        if request.favorite_dishes:
            favorite_dishes = request.favorite_dishes

        user_taste_vec = _user_profile_to_taste_vector(request.user_profile) if request.user_profile else [0.0] * 6

        for m in matches:
            meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
            score = float(m.get("score", 0.0)) if isinstance(m, dict) else float(getattr(m, "score", 0.0))
            menu_items = meta.get("menu_items") or []
            if not _allergy_filter(menu_items, allergies):
                continue
            taste_vec = meta.get("taste_vector") or [0.0] * 6
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
                "location": meta.get("location"),
                "coordinates": meta.get("coordinates"),
                "menu_items": menu_items,
                "popular_dishes": meta.get("popular_dishes"),
                "taste_vector": taste_vec,
                "photos": meta.get("photos"),
                "menu_url": meta.get("menu_url"),
                "score": combined
            })

        ranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    except Exception:
        ranked = []

    ai_json["menu_buddy"] = {
        "seed_restaurants": restaurants,
        "recommendations": ranked
    }

    return ai_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

