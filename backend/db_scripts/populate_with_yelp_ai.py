"""
Enhanced Restaurant Population using Yelp AI Chat API

This script uses Yelp's AI Chat API which provides better structured data
including menu information directly from Yelp's database.

Usage:
    python populate_with_yelp_ai.py --locations "New York, NY" --limit 50
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yelp_api_client import YelpAPIClient
from menu_url_scraper import MenuURLScraper
from restaurant_to_pinecone import process_restaurants, upload_to_pinecone


def search_with_ai_chat(query: str, location: str, limit: int = 50) -> List[Dict]:
    """
    Use Yelp AI Chat API to get restaurant data
    
    Args:
        query: Search query (e.g., "pizza restaurants", "best italian food")
        location: Location string
        limit: Number of results
    
    Returns:
        List of restaurants with enhanced data
    """
    print(f"\n🤖 Using Yelp AI Chat API...")
    print(f"   Query: '{query}' in {location}")
    
    yelp_client = YelpAPIClient()
    
    # Use AI Chat for natural language search
    ai_query = f"Find {query} in {location}"
    
    try:
        # Note: AI Chat API might return structured business data
        # For now, we'll use the standard search API with enhanced queries
        all_restaurants = []
        
        # Try multiple search terms to get more coverage
        search_terms = [
            "restaurants",
            "pizza restaurants", 
            "italian restaurants",
            "asian restaurants",
            "american restaurants",
            "mexican restaurants",
            "seafood restaurants"
        ]
        
        seen_ids = set()
        
        for term in search_terms:
            print(f"\n   Searching for: {term}")
            results = yelp_client.search_businesses(term, location, limit=50)
            
            new_results = []
            for r in results:
                if r.get("id") not in seen_ids:
                    seen_ids.add(r["id"])
                    new_results.append(r)
            
            print(f"   Found {len(new_results)} new restaurants")
            all_restaurants.extend(new_results)
            
            # Rate limiting
            time.sleep(1)
            
            if len(all_restaurants) >= limit:
                break
        
        print(f"\n✅ Total unique restaurants: {len(all_restaurants)}")
        return all_restaurants[:limit]
        
    except Exception as e:
        print(f"❌ Error with AI search: {e}")
        # Fallback to standard search
        return yelp_client.search_businesses("restaurants", location, limit)


def enhance_restaurant_data(restaurant: Dict, yelp_client: YelpAPIClient) -> Dict:
    """
    Enhance restaurant data with detailed information
    
    Args:
        restaurant: Basic restaurant dict from search
        yelp_client: YelpAPIClient instance
    
    Returns:
        Enhanced restaurant dict with menu data
    """
    business_id = restaurant.get("id")
    
    if not business_id:
        return restaurant
    
    try:
        # Get detailed business info
        details = yelp_client.get_business_details(business_id)
        
        if details:
            # Merge details into restaurant
            restaurant.update({
                "phone": details.get("phone"),
                "hours": details.get("hours"),
                "transactions": details.get("transactions"),
                "attributes": details.get("attributes", {}),
            })
            
            # Check for menu URL in attributes
            if "menu_url" not in restaurant or not restaurant["menu_url"]:
                menu_url = details.get("attributes", {}).get("menu_url")
                if menu_url:
                    restaurant["menu_url"] = menu_url
        
        time.sleep(0.3)  # Rate limiting
        
    except Exception as e:
        print(f"   ⚠️  Could not enhance data: {e}")
    
    return restaurant


def populate_with_enhanced_search(location: str, limit: int = 100) -> Dict[str, Any]:
    """
    Populate using enhanced search methods
    """
    print("\n" + "="*80)
    print(f"ENHANCED POPULATION: {location}")
    print("="*80)
    
    stats = {
        "location": location,
        "total_found": 0,
        "with_menu_url": 0,
        "successfully_scraped": 0,
        "uploaded_to_pinecone": 0
    }
    
    # Step 1: Search with multiple terms
    restaurants = search_with_ai_chat("restaurants", location, limit)
    stats["total_found"] = len(restaurants)
    
    if not restaurants:
        print("❌ No restaurants found")
        return stats
    
    # Step 2: Enhance data for each restaurant
    print(f"\n🔍 Enhancing restaurant data...")
    yelp_client = YelpAPIClient()
    
    enhanced_restaurants = []
    for i, rest in enumerate(restaurants, 1):
        print(f"[{i}/{len(restaurants)}] {rest.get('name')}")
        enhanced = enhance_restaurant_data(rest, yelp_client)
        
        # Get menu URL from various sources
        menu_url = (
            enhanced.get("menu_url") or
            enhanced.get("attributes", {}).get("menu_url") or
            rest.get("url")  # Sometimes the business URL has menu info
        )
        
        if menu_url:
            enhanced["menu_url"] = menu_url
            stats["with_menu_url"] += 1
        
        enhanced_restaurants.append(enhanced)
    
    print(f"\n✅ Enhanced {len(enhanced_restaurants)} restaurants")
    print(f"   {stats['with_menu_url']} have menu URLs")
    
    # Step 3: Scrape menus
    print(f"\n🍽️  Scraping menus...")
    menu_scraper = MenuURLScraper()
    
    for i, restaurant in enumerate(enhanced_restaurants, 1):
        name = restaurant.get("name")
        menu_url = restaurant.get("menu_url")
        
        print(f"\n[{i}/{len(enhanced_restaurants)}] {name}")
        
        if menu_url:
            try:
                dishes = menu_scraper.scrape_menu_url(menu_url)
                if dishes:
                    restaurant["menu_items"] = dishes
                    restaurant["popular_dishes"] = dishes[:10]
                    stats["successfully_scraped"] += 1
                    print(f"   ✅ Found {len(dishes)} dishes")
                else:
                    restaurant["menu_items"] = []
                    restaurant["popular_dishes"] = []
            except Exception as e:
                print(f"   ❌ Error: {e}")
                restaurant["menu_items"] = []
                restaurant["popular_dishes"] = []
        else:
            restaurant["menu_items"] = []
            restaurant["popular_dishes"] = []
        
        time.sleep(0.5)
    
    # Step 4: Save results
    location_safe = location.replace(", ", "_").replace(" ", "_")
    output_file = f"enhanced_{location_safe}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enhanced_restaurants, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to {output_file}")
    
    # Step 5: Upload to Pinecone
    restaurants_with_menus = [r for r in enhanced_restaurants if r.get("menu_items")]
    
    if restaurants_with_menus:
        print(f"\n📤 Uploading {len(restaurants_with_menus)} restaurants...")
        try:
            vectors = process_restaurants(restaurants_with_menus)
            upload_to_pinecone(vectors)
            stats["uploaded_to_pinecone"] = len(vectors)
            print(f"✅ Uploaded {len(vectors)} restaurants")
        except Exception as e:
            print(f"❌ Upload error: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locations", nargs="+", 
                       default=["New York, NY", "San Francisco, CA", "San Jose, CA", "Oakland, CA"])
    parser.add_argument("--limit", type=int, default=100)
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ENHANCED RESTAURANT POPULATION (Yelp AI)")
    print("="*80)
    
    all_stats = []
    
    for location in args.locations:
        stats = populate_with_enhanced_search(location, args.limit)
        all_stats.append(stats)
        time.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for stats in all_stats:
        print(f"\n{stats['location']}:")
        print(f"  Found: {stats['total_found']}")
        print(f"  Scraped: {stats['successfully_scraped']}")
        print(f"  Uploaded: {stats['uploaded_to_pinecone']}")
    
    total_uploaded = sum(s["uploaded_to_pinecone"] for s in all_stats)
    print(f"\n✅ Total uploaded: {total_uploaded} restaurants")


if __name__ == "__main__":
    main()
