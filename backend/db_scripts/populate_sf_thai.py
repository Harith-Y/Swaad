"""
Populate San Francisco Thai Restaurants ONLY
Quick targeted population for User 3's location
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from master_populate import RestaurantPopulator
from restaurant_to_pinecone import upload_to_pinecone, create_pinecone_vector, get_embedding_model, load_ingredient_flavor_map


def main():
    """Populate SF Thai restaurants only"""
    
    print("=" * 80)
    print("🍜 SAN FRANCISCO THAI RESTAURANT POPULATION")
    print("=" * 80)
    print("\nTargeted population for SF Thai restaurants")
    print("This will be FAST (~5-10 minutes)\n")
    
    # Just SF, just Thai
    locations = ["San Francisco, CA"]
    
    # Create custom populator with Thai-only search
    populator = RestaurantPopulator()
    
    print(f"⏳ Starting Thai restaurant search in SF...")
    
    try:
        # Override search to only Thai
        restaurants = []
        
        # Search Thai restaurants
        thai_results = populator.yelp_client.search_businesses(
            "thai restaurants", 
            "San Francisco, CA", 
            limit=50
        )
        
        if not thai_results:
            print("\n⚠️  No results from Yelp (likely rate limit hit)")
            print("   Solution: Wait 10-15 minutes and run again")
            print("   The rate limit will reset automatically.\n")
            return
        
        print(f"\n✅ Found {len(thai_results)} Thai restaurants in SF")
        
        for rest in thai_results:
            rest_id = rest.get("id")
            if rest_id and rest_id not in populator.seen_ids:
                populator.seen_ids.add(rest_id)
                
                # Normalize restaurant data
                normalized = {
                    "id": rest_id,
                    "name": rest.get("name"),
                    "url": rest.get("url"),
                    "location": rest.get("location"),
                    "coordinates": rest.get("coordinates"),
                    "price": rest.get("price"),
                    "rating": rest.get("rating", 0.0),
                    "review_count": rest.get("review_count", 0),
                    "categories": rest.get("categories", []),
                    "cuisine_types": [cat["title"] for cat in rest.get("categories", [])],
                    "menu_url": None
                }
                
                # Try to get menu URL
                if rest.get("url"):
                    try:
                        details = populator.yelp_client.get_business_details(rest_id)
                        if details:
                            # Extract menu URL from attributes or hours
                            for attr in details.get("attributes", {}).values():
                                if isinstance(attr, str) and ("menu" in attr.lower() or "http" in attr):
                                    normalized["menu_url"] = attr
                                    break
                    except:
                        pass
                
                restaurants.append(normalized)
        
        print(f"\n🍽️  Scraping menus for {len(restaurants)} restaurants...")
        
        # Scrape and upload
        enhanced = populator.enhance_and_scrape(restaurants)
        
        if enhanced:
            print(f"\n� Processing {len(enhanced)} restaurants for upload...")
            
            # Initialize embedding model and ingredient map
            print("   Loading embedding model...")
            get_embedding_model()
            print("   Loading ingredient-flavor map...")
            load_ingredient_flavor_map()
            
            # Transform restaurants to Pinecone vectors
            vectors = []
            for rest in enhanced:
                try:
                    # Ensure required fields exist
                    if not rest.get("id"):
                        continue
                    
                    # Normalize data structure
                    restaurant_data = {
                        "id": rest["id"],
                        "name": rest.get("name", ""),
                        "avg_rating": rest.get("avg_rating", rest.get("rating", 0.0)),
                        "price_range": rest.get("price"),
                        "cuisine_types": rest.get("cuisine_types", []),
                        "location": rest.get("location", {}),
                        "menu_items": rest.get("menu_items", []),
                        "popular_dishes": rest.get("popular_dishes", []),
                        "taste_vector": [0.0] * 6  # Will be calculated in create_pinecone_vector
                    }
                    
                    vector = create_pinecone_vector(restaurant_data)
                    if vector and vector.get("values"):
                        vectors.append(vector)
                    else:
                        print(f"   ⚠️  Skipping {rest.get('name')}: no embedding generated")
                        
                except Exception as e:
                    print(f"   ⚠️  Error processing {rest.get('name', 'unknown')}: {e}")
                    continue
            
            if vectors:
                print(f"\n📤 Uploading {len(vectors)} restaurants to Pinecone...")
                upload_result = upload_to_pinecone(vectors)
                print(f"✅ Upload complete!")
                print(f"\n🎉 SUCCESS!")
                print(f"   Thai restaurants uploaded: {len(vectors)}")
                print(f"   User 3 (dummy3) can now get Thai curry in SF!")
            else:
                print(f"\n❌ No valid vectors created from {len(enhanced)} restaurants")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
