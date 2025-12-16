"""
Check ratings specifically for pizza restaurants
"""
import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment
load_dotenv()

def check_pizza_ratings():
    """Check ratings for pizza restaurants"""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "menu-buddy")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Create pizza embedding
    print("🍕 Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pizza_vec = model.encode("pizza").tolist()
    
    # Query for pizza restaurants
    print("🔍 Searching for pizza restaurants...\n")
    results = index.query(
        vector=pizza_vec,
        top_k=50,
        include_metadata=True,
        namespace="restaurants"
    )
    
    print(f"Found {len(results.matches)} pizza-related restaurants\n")
    print("=" * 100)
    print(f"{'Restaurant':<40} {'City':<15} {'Rating':<10} {'Has Pizza?'}")
    print("=" * 100)
    
    zero_rating = 0
    has_rating = 0
    
    for i, match in enumerate(results.matches, 1):
        meta = match.metadata
        name = meta.get("name", "Unknown")
        rating = meta.get("avg_rating", 0.0)
        menu_items = meta.get("menu_items", [])
        
        city = "Unknown"
        try:
            location_json = meta.get("location_json", "{}")
            if location_json:
                location = json.loads(location_json)
                city = location.get("city", "Unknown")
        except:
            pass
        
        # Check if pizza in name or menu
        has_pizza_in_name = "pizza" in name.lower()
        has_pizza_in_menu = any("pizza" in item.lower() for item in menu_items)
        pizza_status = ""
        if has_pizza_in_name:
            pizza_status = "Name"
        elif has_pizza_in_menu:
            pizza_status = "Menu"
        
        if rating == 0.0:
            zero_rating += 1
        else:
            has_rating += 1
        
        rating_display = f"{rating}/5" if rating > 0 else "NO RATING"
        
        if i <= 20:  # Show first 20
            print(f"{name:<40} {city:<15} {rating_display:<10} {pizza_status}")
    
    print("=" * 100)
    print(f"\n📊 Summary:")
    print(f"   Total: {len(results.matches)}")
    print(f"   With rating: {has_rating} ({has_rating/len(results.matches)*100:.1f}%)")
    print(f"   No rating (0.0): {zero_rating} ({zero_rating/len(results.matches)*100:.1f}%)")
    
    if zero_rating > 0:
        print(f"\n⚠️  {zero_rating} pizza restaurants have 0.0 rating!")
        print("   These need to be re-scraped from Yelp API with proper rating field.")


if __name__ == "__main__":
    check_pizza_ratings()
