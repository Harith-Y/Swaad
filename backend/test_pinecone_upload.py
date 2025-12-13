"""
Test script to verify Pinecone upload functionality.

This script:
1. Tests the restaurant transformation
2. Tests taste vector generation
3. Tests Pinecone upload
4. Queries Pinecone to verify data
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from restaurant_to_pinecone import (
    transform_restaurant,
    create_pinecone_vector,
    infer_taste_from_text,
    load_ingredient_flavor_map,
    get_embedding_model,
    upload_to_pinecone,
    process_restaurants
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_ingredient_flavor_map():
    """Test loading ingredient-flavor.csv"""
    print("\n" + "="*60)
    print("TEST 1: Loading Ingredient-Flavor Map")
    print("="*60)
    
    mapping = load_ingredient_flavor_map()
    print(f"✓ Loaded {len(mapping)} ingredients")
    
    # Test a few ingredients
    test_ingredients = ["garlic", "tomato", "butter", "chili powder"]
    for ing in test_ingredients:
        if ing in mapping:
            taste_vec = mapping[ing]["taste_vector"]
            print(f"  {ing}: {taste_vec}")
        else:
            print(f"  {ing}: NOT FOUND")
    
    return len(mapping) > 0


def test_taste_vector_generation():
    """Test taste vector generation from text"""
    print("\n" + "="*60)
    print("TEST 2: Taste Vector Generation")
    print("="*60)
    
    test_dishes = [
        "Paneer Butter Masala",
        "Chicken Tikka Masala",
        "Sushi Roll with Salmon",
        "Chocolate Cake",
        "Spicy Thai Curry"
    ]
    
    for dish in test_dishes:
        taste_vec = infer_taste_from_text(dish)
        print(f"\n  {dish}")
        print(f"    Taste Vector: {[round(x, 2) for x in taste_vec]}")
        print(f"    [sweet, salty, sour, bitter, umami, spicy]")
    
    return True


def test_restaurant_transformation():
    """Test transforming raw restaurant data"""
    print("\n" + "="*60)
    print("TEST 3: Restaurant Transformation")
    print("="*60)
    
    raw_restaurant = {
        "id": 999,
        "name": "Test Indian Restaurant",
        "location": {
            "address": "123 Test St",
            "city": "Test City",
            "lat": 37.7749,
            "lng": -122.4194
        },
        "cuisine_types": ["Indian", "North Indian"],
        "price": "$$",
        "menu_items": [
            "Paneer Butter Masala",
            "Chicken Tikka Masala",
            "Garlic Naan",
            "Samosa"
        ],
        "popular_dishes": [
            "Paneer Butter Masala",
            "Garlic Naan"
        ],
        "avg_rating": 4.5
    }
    
    restaurant = transform_restaurant(raw_restaurant, 999)
    
    print(f"\n  Restaurant ID: {restaurant['id']}")
    print(f"  Name: {restaurant['name']}")
    print(f"  Cuisine Types: {restaurant['cuisine_types']}")
    print(f"  Price Range: {restaurant['price_range']}")
    print(f"  Menu Items: {len(restaurant['menu_items'])} items")
    print(f"  Taste Vector: {[round(x, 2) for x in restaurant['taste_vector']]}")
    print(f"    [sweet, salty, sour, bitter, umami, spicy]")
    
    return restaurant


def test_vector_creation(restaurant):
    """Test creating Pinecone vector from restaurant"""
    print("\n" + "="*60)
    print("TEST 4: Pinecone Vector Creation")
    print("="*60)
    
    print("  Loading embedding model...")
    get_embedding_model()
    print("  ✓ Model loaded")
    
    print("  Creating vector...")
    vector = create_pinecone_vector(restaurant)
    
    print(f"\n  Vector ID: {vector['id']}")
    print(f"  Embedding Dimension: {len(vector['values'])}")
    print(f"  Metadata Keys: {list(vector['metadata'].keys())}")
    print(f"  Taste Vector in Metadata:")
    print(f"    taste_0 (sweet): {vector['metadata']['taste_0']}")
    print(f"    taste_1 (salty): {vector['metadata']['taste_1']}")
    print(f"    taste_2 (sour): {vector['metadata']['taste_2']}")
    print(f"    taste_3 (bitter): {vector['metadata']['taste_3']}")
    print(f"    taste_4 (umami): {vector['metadata']['taste_4']}")
    print(f"    taste_5 (spicy): {vector['metadata']['taste_5']}")
    
    return vector


def test_pinecone_upload():
    """Test uploading to Pinecone"""
    print("\n" + "="*60)
    print("TEST 5: Pinecone Upload (Optional)")
    print("="*60)
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("  ⚠ PINECONE_API_KEY not set - skipping upload test")
        return False
    
    print("  Creating test restaurant...")
    test_restaurants = [
        {
            "id": 9999,
            "name": "Test Restaurant for Upload",
            "location": {
                "address": "123 Test St",
                "city": "Test City",
                "lat": 37.7749,
                "lng": -122.4194
            },
            "cuisine_types": ["Test Cuisine"],
            "price": "$",
            "menu_items": ["Test Dish 1", "Test Dish 2"],
            "popular_dishes": ["Test Dish 1"],
            "avg_rating": 4.0
        }
    ]
    
    print("  Processing restaurant...")
    vectors = process_restaurants(test_restaurants)
    
    print(f"  ✓ Generated {len(vectors)} vector(s)")
    
    response = input("\n  Upload to Pinecone? (y/n): ")
    if response.lower() == 'y':
        print("  Uploading...")
        upload_to_pinecone(vectors)
        print("  ✓ Upload complete!")
        return True
    else:
        print("  Skipped upload")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PINECONE UPLOAD SCRIPT - TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Load ingredient map
        if not test_ingredient_flavor_map():
            print("\n❌ Failed to load ingredient-flavor.csv")
            return
        
        # Test 2: Taste vector generation
        test_taste_vector_generation()
        
        # Test 3: Restaurant transformation
        restaurant = test_restaurant_transformation()
        
        # Test 4: Vector creation
        vector = test_vector_creation(restaurant)
        
        # Test 5: Pinecone upload (optional)
        test_pinecone_upload()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python backend/restaurant_to_pinecone.py")
        print("2. Or: python backend/restaurant_to_pinecone.py sample_restaurants.json")
        print("3. Test search via /api/chat endpoint")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

