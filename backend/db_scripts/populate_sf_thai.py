"""
Populate San Francisco Thai Restaurants ONLY
Quick targeted population for User 3's location
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from master_populate import RestaurantPopulator
from restaurant_to_pinecone import upload_to_pinecone


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
            print(f"\n📤 Uploading {len(enhanced)} restaurants to Pinecone...")
            upload_result = upload_to_pinecone(enhanced)
            print(f"✅ Upload complete!")
        
        print(f"\n🎉 SUCCESS!")
        print(f"   Thai restaurants populated: {len(enhanced)}")
        print(f"   User 3 (dummy3) can now get Thai curry in SF!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
