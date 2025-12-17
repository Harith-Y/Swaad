"""
Quick check for Chicago restaurants in Pinecone
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def check_chicago_restaurants():
    """Check if Chicago restaurants exist in Pinecone"""
    print("Checking for Chicago restaurants in Pinecone...\n")
    
    try:
        pc_index = get_pinecone_index()
        
        # Query for Chicago restaurants (using dummy vector)
        dummy_vector = [0.0] * 384
        
        # Try getting all results
        results = pc_index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            namespace="restaurants"
        )
        print(f"Query returned {len(results.get('matches', []))} total results\n")
        
        # Filter for Chicago
        chicago_restaurants = []
        all_cities = set()
        
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            
            # Collect all cities
            city = meta.get("city", "")
            if city:
                all_cities.add(city)
            
            # Check for Chicago
            if isinstance(city, str) and "chicago" in city.lower():
                chicago_restaurants.append({
                    "name": meta.get("name"),
                    "city": city,
                    "cuisine": meta.get("cuisine_types"),
                    "dishes": len(meta.get("menu_items", []))
                })
        
        print(f"All cities found: {sorted(all_cities)}\n")
        
        print(f"✅ Found {len(chicago_restaurants)} Chicago restaurants\n")
        
        if chicago_restaurants:
            print("Sample Chicago restaurants:")
            for rest in chicago_restaurants[:10]:
                print(f"  - {rest['name']}")
                print(f"    City: {rest.get('city', 'N/A')}")
                print(f"    Cuisine: {rest['cuisine']}")
                print(f"    Dishes: {rest['dishes']}")
                print()
        else:
            print("❌ No Chicago restaurants found in database")
            print("Run populate_chicago.py to add them.\n")
        
        return len(chicago_restaurants)
        
    except Exception as e:
        print(f"Error: {e}")
        return 0


if __name__ == "__main__":
    check_chicago_restaurants()
