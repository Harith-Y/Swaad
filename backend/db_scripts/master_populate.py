"""
Master Restaurant Population Script

This script orchestrates comprehensive restaurant and dish population
across all target locations with proper error handling and reporting.

Features:
- Multiple search strategies (general, category-specific, cuisine-specific)
- Intelligent deduplication
- Progress tracking
- Comprehensive error handling
- Detailed reporting

Usage:
    python master_populate.py
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yelp_api_client import YelpAPIClient
from menu_url_scraper import MenuURLScraper
from restaurant_to_pinecone import process_restaurants, upload_to_pinecone


class RestaurantPopulator:
    """Comprehensive restaurant population manager"""
    
    def __init__(self):
        self.yelp_client = YelpAPIClient()
        self.menu_scraper = MenuURLScraper()
        self.seen_ids: Set[str] = set()
        self.all_restaurants: List[Dict] = []
        self.stats = {
            "total_searched": 0,
            "unique_found": 0,
            "with_menu_url": 0,
            "successfully_scraped": 0,
            "uploaded": 0,
            "errors": 0
        }
    
    def search_location_comprehensive(self, location: str, limit_per_category: int = 50) -> List[Dict]:
        """
        Search a location using multiple strategies
        
        Args:
            location: Location string
            limit_per_category: Results per search term
        
        Returns:
            List of unique restaurants
        """
        print(f"\n{'='*80}")
        print(f"📍 COMPREHENSIVE SEARCH: {location}")
        print(f"{'='*80}")
        
        # Multiple search strategies
        search_terms = [
            "restaurants",
            "pizza restaurants",
            "italian restaurants",
            "chinese restaurants",
            "mexican restaurants",
            "japanese restaurants",
            "thai restaurants",
            "indian restaurants",
            "american restaurants",
            "seafood restaurants",
            "burger restaurants",
            "steakhouse",
            "sushi restaurants",
            "breakfast restaurants",
            "brunch restaurants"
        ]
        
        location_restaurants = []
        location_ids = set()
        
        for term in search_terms:
            print(f"\n🔍 Searching: '{term}' in {location}")
            
            try:
                results = self.yelp_client.search_businesses(term, location, limit_per_category)
                self.stats["total_searched"] += len(results)
                
                new_count = 0
                for rest in results:
                    rest_id = rest.get("id")
                    if rest_id and rest_id not in location_ids and rest_id not in self.seen_ids:
                        location_ids.add(rest_id)
                        self.seen_ids.add(rest_id)
                        
                        # Extract and normalize data from Yelp response
                        location_data = rest.get("location", {})
                        categories = rest.get("categories", [])
                        
                        normalized_rest = {
                            "id": rest_id,
                            "name": rest.get("name", ""),
                            "avg_rating": rest.get("rating", 0.0),  # PRESERVE RATING from search
                            "location": {
                                "address": ", ".join(location_data.get("display_address", [])),
                                "city": location_data.get("city", ""),
                                "lat": rest.get("coordinates", {}).get("latitude"),
                                "lng": rest.get("coordinates", {}).get("longitude")
                            },
                            "cuisine_types": [cat.get("title") for cat in categories],
                            "price": rest.get("price"),
                            "phone": rest.get("phone", ""),
                            "url": rest.get("url", ""),
                            "menu_url": rest.get("attributes", {}).get("menu_url") if rest.get("attributes") else None,
                        }
                        
                        location_restaurants.append(normalized_rest)
                        new_count += 1
                
                print(f"   Found {len(results)} results, {new_count} new")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.stats["errors"] += 1
        
        print(f"\n✅ Total unique restaurants for {location}: {len(location_restaurants)}")
        self.stats["unique_found"] += len(location_restaurants)
        
        return location_restaurants
    
    def enhance_and_scrape(self, restaurants: List[Dict]) -> List[Dict]:
        """
        Enhance restaurant data and scrape menus
        
        Args:
            restaurants: List of basic restaurant dicts
        
        Returns:
            List of enhanced restaurants with menu data
        """
        print(f"\n🔍 Enhancing {len(restaurants)} restaurants...")
        
        enhanced = []
        
        for i, rest in enumerate(restaurants, 1):
            name = rest.get("name")
            rest_id = rest.get("id")
            
            print(f"\n[{i}/{len(restaurants)}] {name}")
            
            # Get detailed info
            try:
                details = self.yelp_client.get_business_details(rest_id)
                if details:
                    rest["phone"] = details.get("phone", rest.get("phone", ""))
                    rest["hours"] = details.get("hours")
                    
                    # Preserve or update rating
                    if "avg_rating" not in rest or rest["avg_rating"] == 0.0:
                        rest["avg_rating"] = details.get("rating", 0.0)
                    
                    # Get menu URL from attributes
                    menu_url = details.get("attributes", {}).get("menu_url")
                    if menu_url:
                        rest["menu_url"] = menu_url
                        self.stats["with_menu_url"] += 1
                
                time.sleep(0.3)
                
            except Exception as e:
                print(f"   ⚠️  Enhancement failed: {e}")
            
            # Scrape menu if URL exists
            menu_url = rest.get("menu_url")
            if menu_url:
                print(f"   🍽️  Scraping menu...")
                try:
                    dishes = self.menu_scraper.scrape_menu_url(menu_url)
                    if dishes:
                        rest["menu_items"] = dishes
                        rest["popular_dishes"] = dishes[:10]
                        self.stats["successfully_scraped"] += 1
                        print(f"   ✅ Found {len(dishes)} dishes")
                    else:
                        rest["menu_items"] = []
                        rest["popular_dishes"] = []
                        print(f"   ⚠️  No dishes extracted")
                except Exception as e:
                    print(f"   ❌ Scraping error: {e}")
                    rest["menu_items"] = []
                    rest["popular_dishes"] = []
                    self.stats["errors"] += 1
            else:
                rest["menu_items"] = []
                rest["popular_dishes"] = []
                print(f"   ℹ️  No menu URL")
            
            enhanced.append(rest)
            time.sleep(0.5)
        
        return enhanced
    
    def populate_locations(self, locations: List[str], limit_per_category: int = 50):
        """
        Main method to populate multiple locations
        
        Args:
            locations: List of location strings
            limit_per_category: Results per search category
        """
        print(f"\n{'='*80}")
        print(f"🚀 MASTER RESTAURANT POPULATION")
        print(f"{'='*80}")
        print(f"\nLocations: {', '.join(locations)}")
        print(f"Categories per location: 15")
        print(f"Limit per category: {limit_per_category}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Process each location
        for location in locations:
            # Search
            location_restaurants = self.search_location_comprehensive(location, limit_per_category)
            
            # Enhance and scrape
            if location_restaurants:
                enhanced = self.enhance_and_scrape(location_restaurants)
                self.all_restaurants.extend(enhanced)
                
                # Upload this location's restaurants immediately
                restaurants_with_menus = [r for r in enhanced if r.get("menu_items")]
                
                if restaurants_with_menus:
                    print(f"\n📤 Uploading {len(restaurants_with_menus)} restaurants from {location} to Pinecone...")
                    try:
                        vectors = process_restaurants(restaurants_with_menus)
                        upload_to_pinecone(vectors)
                        self.stats["uploaded"] += len(vectors)
                        print(f"✅ Successfully uploaded {len(vectors)} restaurants from {location}!")
                    except Exception as e:
                        print(f"❌ Upload error for {location}: {e}")
                        self.stats["errors"] += 1
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"\n⚠️  No restaurants with menus from {location}")
            
            # Brief pause between locations
            print(f"\n⏸️  Pausing before next location...")
            time.sleep(3)
        
        # Save all results
        print(f"\n💾 Saving results...")
        output_file = f"master_restaurants_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.all_restaurants, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.all_restaurants)} restaurants to {output_file}")
        
        # Final report
        self.print_final_report()
    
    def print_final_report(self):
        """Print comprehensive final report"""
        print(f"\n{'='*80}")
        print(f"📊 FINAL REPORT")
        print(f"{'='*80}")
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📈 Statistics:")
        print(f"   Total searches performed: {self.stats['total_searched']}")
        print(f"   Unique restaurants found: {self.stats['unique_found']}")
        print(f"   Restaurants with menu URLs: {self.stats['with_menu_url']}")
        print(f"   Successfully scraped menus: {self.stats['successfully_scraped']}")
        print(f"   Uploaded to Pinecone: {self.stats['uploaded']}")
        print(f"   Errors encountered: {self.stats['errors']}")
        
        if self.stats['unique_found'] > 0:
            scrape_rate = (self.stats['successfully_scraped'] / self.stats['unique_found']) * 100
            upload_rate = (self.stats['uploaded'] / self.stats['unique_found']) * 100
            print(f"\n📊 Success Rates:")
            print(f"   Menu scraping: {scrape_rate:.1f}%")
            print(f"   Upload rate: {upload_rate:.1f}%")
        
        # Dish statistics
        total_dishes = sum(len(r.get("menu_items", [])) for r in self.all_restaurants)
        avg_dishes = total_dishes / len(self.all_restaurants) if self.all_restaurants else 0
        
        print(f"\n🍽️  Dish Statistics:")
        print(f"   Total dishes collected: {total_dishes}")
        print(f"   Average dishes per restaurant: {avg_dishes:.1f}")
        
        print(f"\n✅ Population complete!")


def main():
    """Main execution"""
    
    # Target locations
    locations = [
        "San Francisco, CA",
        "San Jose, CA",
        "Oakland, CA",
        "New York, NY",
        "Brooklyn, NY",
        "Manhattan, NY"
    ]
    
    # Create populator and run
    populator = RestaurantPopulator()
    populator.populate_locations(locations, limit_per_category=50)


if __name__ == "__main__":
    main()
