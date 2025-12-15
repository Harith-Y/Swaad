"""
Yelp API Client - Uses Yelp Fusion API to get business details with menu URLs

This script:
1. Uses Yelp AI Chat API to search for businesses
2. Extracts business.attributes.menuUrl from the response
3. Returns structured business data ready for menu scraping

Requirements:
    pip install requests python-dotenv

Environment Variables:
    YELP_API_KEY=your_yelp_api_key_here
"""

import os
import json
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class YelpAPIClient:
    def __init__(self):
        self.api_key = os.getenv("YELP_API_KEY")
        if not self.api_key:
            raise ValueError("YELP_API_KEY not found in environment variables")
        
        self.base_url = "https://api.yelp.com/v3"
        self.ai_url = "https://api.yelp.com/ai/chat/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def search_businesses(self, term: str, location: str, limit: int = 20) -> List[Dict]:
        """
        Search for businesses using Yelp Fusion API
        
        Args:
            term: Search term (e.g., "restaurants", "italian food")
            location: Location (e.g., "San Francisco", "New York")
            limit: Number of results (max 50)
        
        Returns:
            List of business dictionaries
        """
        url = f"{self.base_url}/businesses/search"
        params = {
            "term": term,
            "location": location,
            "limit": min(limit, 50)
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("businesses", [])
        except Exception as e:
            print(f"❌ Error searching businesses: {e}")
            return []
    
    def get_business_details(self, business_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific business
        
        Args:
            business_id: Yelp business ID
        
        Returns:
            Business details dictionary with menuUrl if available
        """
        url = f"{self.base_url}/businesses/{business_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting business details: {e}")
            return None
    
    def ai_chat_search(self, query: str, latitude: float = None, longitude: float = None, 
                       chat_id: str = None) -> Dict:
        """
        Use Yelp AI Chat API for natural language search
        
        Args:
            query: Natural language query (e.g., "Find Thai restaurants near me")
            latitude: User's latitude for location-based results
            longitude: User's longitude for location-based results
            chat_id: Chat ID for multi-turn conversations (None for new conversation)
        
        Returns:
            AI Chat response with businesses and structured data
        """
        payload = {
            "query": query
        }
        
        if chat_id:
            payload["chat_id"] = chat_id
        
        if latitude and longitude:
            payload["user_context"] = {
                "latitude": latitude,
                "longitude": longitude
            }
        
        try:
            response = requests.post(self.ai_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error with AI chat search: {e}")
            return {}
    
    def extract_menu_urls(self, businesses: List[Dict]) -> List[Dict]:
        """
        Extract menu URLs and relevant info from business list
        
        Args:
            businesses: List of business dictionaries from Yelp API
        
        Returns:
            List of dictionaries with business info and menu URLs
        """
        results = []
        
        for business in businesses:
            business_id = business.get("id")
            name = business.get("name")
            
            # Get detailed info to access attributes.menuUrl
            details = self.get_business_details(business_id)
            if not details:
                continue
            
            menu_url = details.get("attributes", {}).get("menu_url") or \
                      details.get("menu_url") or \
                      None
            
            location = details.get("location", {})
            categories = details.get("categories", [])
            
            restaurant_data = {
                "id": business_id,
                "name": name,
                "menu_url": menu_url,
                "location": {
                    "address": ", ".join(location.get("display_address", [])),
                    "city": location.get("city", ""),
                    "lat": details.get("coordinates", {}).get("latitude"),
                    "lng": details.get("coordinates", {}).get("longitude")
                },
                "cuisine_types": [cat.get("title") for cat in categories],
                "price": details.get("price"),
                "avg_rating": details.get("rating", 0.0),
                "phone": details.get("phone", ""),
                "url": details.get("url", "")
            }
            
            results.append(restaurant_data)
        
        return results
    
    def search_and_get_menu_urls(self, term: str, location: str, limit: int = 20) -> List[Dict]:
        """
        Search for businesses and extract their menu URLs in one call
        
        Args:
            term: Search term
            location: Location
            limit: Number of results
        
        Returns:
            List of restaurant data with menu URLs
        """
        print(f"🔍 Searching for '{term}' in {location}...")
        businesses = self.search_businesses(term, location, limit)
        print(f"   Found {len(businesses)} businesses")
        
        print(f"📋 Extracting menu URLs...")
        restaurants = self.extract_menu_urls(businesses)
        
        menu_count = sum(1 for r in restaurants if r.get("menu_url"))
        print(f"   Found {menu_count} restaurants with menu URLs")
        
        return restaurants


def main():
    """Example usage"""
    client = YelpAPIClient()
    
    # Example 1: Search for restaurants and get menu URLs
    print("\n" + "="*60)
    print("YELP API CLIENT - MENU URL EXTRACTION")
    print("="*60)
    
    # Search for restaurants
    restaurants = client.search_and_get_menu_urls(
        term="restaurants",
        location="San Francisco",
        limit=10
    )
    
    # Display results
    print("\n📊 Results:")
    for i, restaurant in enumerate(restaurants, 1):
        print(f"\n{i}. {restaurant['name']}")
        print(f"   Cuisines: {', '.join(restaurant['cuisine_types'])}")
        print(f"   Location: {restaurant['location']['city']}")
        print(f"   Rating: {restaurant['avg_rating']}/5")
        print(f"   Menu URL: {restaurant['menu_url'] or 'Not available'}")
    
    # Save to JSON
    output_file = "restaurants_with_menu_urls.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(restaurants, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(restaurants)} restaurants to {output_file}")
    
    # Example 2: AI Chat search
    print("\n" + "="*60)
    print("AI CHAT SEARCH EXAMPLE")
    print("="*60)
    
    ai_response = client.ai_chat_search(
        query="Find the best Italian restaurants in San Francisco",
        latitude=37.7749,
        longitude=-122.4194
    )
    
    print(f"\nAI Response:")
    print(f"  Chat ID: {ai_response.get('chat_id', 'N/A')}")
    print(f"  Response: {ai_response.get('text', 'N/A')[:200]}...")


if __name__ == "__main__":
    main()
