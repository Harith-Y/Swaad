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
from database import get_db, User, Base, engine
from auth import (
    verify_password, get_password_hash, create_access_token,
    get_user_by_email, get_user_by_username, get_current_user
)
from dotenv import load_dotenv
import os
from groq import Groq
import httpx

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
    df = load_recipes()
    
    user_profile = request.user_profile
    menu_dishes = request.menu_dishes
    categorized_dishes = request.categorized_dishes
    
    # Use categorized dishes if provided, otherwise categorize from flat list
    if categorized_dishes:
        dish_categories = categorized_dishes
    else:
        # Fallback: categorize dishes ourselves
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
    
    # Find recipes matching menu dishes, keeping track of menu dish name -> recipe mapping
    # This preserves the original menu dish names for display
    categorized_menu_dish_mappings = {
        "appetizer": [],  # List of tuples: (menu_dish_name, recipe_dict)
        "mains": [],
        "desserts": []
    }
    
    for category, dishes in dish_categories.items():
        for dish_name in dishes:
            recipe = find_recipe_by_name(dish_name, df)
            if recipe:
                # Check for allergies
                if hasattr(user_profile, 'allergies') and user_profile.allergies:
                    ingredients = recipe.get('ingredients', [])
                    if isinstance(ingredients, str):
                        try:
                            ingredients = ast.literal_eval(ingredients)
                        except:
                            ingredients = []
                    
                    # Simple keyword check for allergies
                    has_allergy = False
                    for allergy in user_profile.allergies:
                        allergy_lower = allergy.lower()
                        for ingredient in ingredients:
                            if allergy_lower in ingredient.lower():
                                has_allergy = True
                                break
                        if has_allergy:
                            break
                    
                    if has_allergy:
                        continue # Skip this recipe

                # Store the mapping: (original menu dish name, matched recipe)
                categorized_menu_dish_mappings[category].append((dish_name, recipe))
    
    # Get recommendations for each category
    recommendations = {
        "appetizer": [],
        "mains": [],
        "desserts": []
    }
    
    for category in ["appetizer", "mains", "desserts"]:
        # Fix Pydantic deprecation: use model_dump() instead of dict()
        user_profile_dict = user_profile.model_dump()[category]
        menu_dish_mappings = categorized_menu_dish_mappings[category]
        
        # Calculate similarity scores, keeping track of menu dish names
        scored_items = []
        for menu_dish_name, recipe in menu_dish_mappings:
            if isinstance(recipe['flavor_profile'], str):
                recipe_profile = ast.literal_eval(recipe['flavor_profile'])
            else:
                recipe_profile = recipe['flavor_profile']
            
            similarity = calculate_similarity(user_profile_dict, recipe_profile)
            # Store: (menu_dish_name, recipe, similarity_score)
            scored_items.append((menu_dish_name, recipe, similarity))
        
        # Sort by similarity and get top 5
        scored_items.sort(key=lambda x: x[2], reverse=True)
        
        for menu_dish_name, recipe, score in scored_items[:5]:
            if isinstance(recipe['ingredients'], str):
                ingredients = ast.literal_eval(recipe['ingredients'])
            else:
                ingredients = recipe['ingredients']
            
            if isinstance(recipe['flavor_profile'], str):
                flavor_profile = ast.literal_eval(recipe['flavor_profile'])
            else:
                flavor_profile = recipe['flavor_profile']
            
            # Use the original menu dish name instead of recipe name
            recommendations[category].append({
                "id": int(recipe['id']),
                "name": menu_dish_name,  # Use menu dish name, not recipe name
                "ingredients": ingredients,
                "flavor_profile": flavor_profile,
                "similarity_score": round(score, 3),
                "category": category
            })
    
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

@app.post("/api/chat")
async def chat_with_yelp(request: ChatRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")
    yelp_api_key = os.getenv("YELP_API_KEY")
    
    if not groq_api_key or not yelp_api_key:
        raise HTTPException(status_code=500, detail="Missing API keys")

    client = Groq(api_key=groq_api_key)
    
    # Prompt for Groq to convert query to Yelp API format
    system_prompt = """
    You are an assistant that converts natural language queries into a JSON body for the Yelp AI Chat API.
    
    The Yelp AI Chat API expects a POST body like this:
    {
        "query": "string",
        "chat_id": "string (optional)",
        "user_context": {
            "latitude": float,
            "longitude": float
        },
        "request_context": {
            "max_results": integer
        }
    }
    
    If the user provides a location (e.g., "in San Francisco"), try to estimate the latitude and longitude and include them in `user_context`. 
    If no location is provided, omit `user_context`.
    Always include the `query` field with the user's original query or a refined version.
    If `chat_id` is provided in the input, include it in the output.
    
    Output ONLY valid JSON.
    """
    
    user_message = f"""
    User Query: {request.query}
    Chat ID: {request.chat_id}
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        yelp_request_body = json.loads(completion.choices[0].message.content)
        
        # Ensure chat_id is passed if it exists in the request and not in the generated body
        if request.chat_id and "chat_id" not in yelp_request_body:
            yelp_request_body["chat_id"] = request.chat_id
            
    except Exception as e:
        print(f"Groq Error: {e}")
        # Fallback: just pass the query directly
        yelp_request_body = {
            "query": request.query
        }
        if request.chat_id:
            yelp_request_body["chat_id"] = request.chat_id

    # Call Yelp API
    yelp_url = "https://api.yelp.com/ai/chat/v2"
    headers = {
        "Authorization": f"Bearer {yelp_api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(yelp_url, json=yelp_request_body, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Yelp API Error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

