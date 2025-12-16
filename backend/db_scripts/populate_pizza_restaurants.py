"""
Populate Pizza Restaurants Specifically

This script focuses on finding and populating pizza restaurants with proper dish data.
It targets pizza-specific searches to ensure we have good coverage.

Usage:
    python populate_pizza_restaurants.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yelp_api_client import YelpAPIClient
from menu_url_scraper import MenuURLScraper
from restaurant_to_pinecone import process_restaurants, upload_to_pinecone


def populate_pizza_restaurants():
    """Populate pizza restaurants for all major cities"""
    
    locations = [
        "New York, NY",
        "Brooklyn, NY", 
        "Manhattan, NY",
        "San Francisco, CA",
        "San Jose, CA",
        "Oakland, CA"
    ]
    
    print("\n" + "="*80)
    print("🍕 PIZZA RESTAURANT POPULATION")
    print("="*80)
    
    yelp_client = YelpAPIClient()
    menu_scraper = MenuURLScraper()
    
    all_pizza_restaurants = []
    seen_ids = set()
    
    for location in locations:
        print(f"\n📍 Searching pizza in {location}...")
        
        # Search with multiple pizza-related terms
        search_terms = ["pizza", "pizzeria", "italian pizza", "new york pizza", "pizza restaurant"]
        
        for term in search_terms:
            try:
                results = yelp_client.search_businesses(term, location, limit=50)
                print(f"   '{term}': {len(results)} results")
                
                for rest in results:
                    rest_id = rest.get("id")
                    if rest_id and rest_id not in seen_ids:
                        seen_ids.add(rest_id)
                        
                        # Enhance with details
                        try:
                            details = yelp_client.get_business_details(rest_id)
                            if details:
                                rest["menu_url"] = details.get("attributes", {}).get("menu_url") or rest.get("url")
                                rest["phone"] = details.get("phone")
                                rest["hours"] = details.get("hours")
                        except:
                            pass
                        
                        all_pizza_restaurants.append(rest)
                        time.sleep(0.3)
                
                time.sleep(1)  # Rate limiting between terms
                
            except Exception as e:
                print(f"   ⚠️  Error searching '{term}': {e}")
    
    print(f"\n✅ Found {len(all_pizza_restaurants)} unique pizza restaurants")
    
    # Scrape menus
    print(f"\n🍽️  Scraping pizza menus...")
    
    successfully_scraped = 0
    
    for i, restaurant in enumerate(all_pizza_restaurants, 1):
        name = restaurant.get("name")
        menu_url = restaurant.get("menu_url")
        
        print(f"\n[{i}/{len(all_pizza_restaurants)}] {name}")
        
        if menu_url:
            try:
                dishes = menu_scraper.scrape_menu_url(menu_url)
                if dishes:
                    restaurant["menu_items"] = dishes
                    restaurant["popular_dishes"] = dishes[:10]
                    successfully_scraped += 1
                    
                    # Count pizza dishes
                    pizza_dishes = [d for d in dishes if "pizza" in d.lower()]
                    print(f"   ✅ {len(dishes)} dishes ({len(pizza_dishes)} pizza dishes)")
                else:
                    restaurant["menu_items"] = []
                    restaurant["popular_dishes"] = []
                    print(f"   ⚠️  No dishes found")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                restaurant["menu_items"] = []
                restaurant["popular_dishes"] = []
        else:
            print(f"   ⚠️  No menu URL")
            restaurant["menu_items"] = []
            restaurant["popular_dishes"] = []
        
        time.sleep(0.5)
    
    print(f"\n✅ Successfully scraped {successfully_scraped}/{len(all_pizza_restaurants)} restaurants")
    
    # Save results
    output_file = "pizza_restaurants.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_pizza_restaurants, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to {output_file}")
    
    # Upload to Pinecone
    restaurants_with_menus = [r for r in all_pizza_restaurants if r.get("menu_items")]
    
    if restaurants_with_menus:
        print(f"\n📤 Uploading {len(restaurants_with_menus)} pizza restaurants to Pinecone...")
        try:
            vectors = process_restaurants(restaurants_with_menus)
            upload_to_pinecone(vectors)
            print(f"✅ Successfully uploaded {len(vectors)} pizza restaurants!")
            
            # Show some stats
            total_dishes = sum(len(r.get("menu_items", [])) for r in restaurants_with_menus)
            total_pizza_dishes = sum(
                len([d for d in r.get("menu_items", []) if "pizza" in d.lower()])
                for r in restaurants_with_menus
            )
            
            print(f"\n📊 Statistics:")
            print(f"   Total pizza restaurants: {len(vectors)}")
            print(f"   Total dishes: {total_dishes}")
            print(f"   Total pizza dishes: {total_pizza_dishes}")
            
        except Exception as e:
            print(f"❌ Error uploading: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No restaurants with menus to upload")
    
    print("\n✅ Pizza population complete! 🍕")


if __name__ == "__main__":
    populate_pizza_restaurants()
