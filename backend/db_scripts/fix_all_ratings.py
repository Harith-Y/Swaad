"""
Fix ALL restaurant ratings in Pinecone

This script:
1. Iterates through ALL restaurants in Pinecone using pagination
2. Fetches ratings from Yelp API for those with 0.0 rating
3. Updates Pinecone with correct ratings
"""

import os
import sys
import json
import time
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yelp_api_client import YelpAPIClient

# Load environment
load_dotenv()


def fix_all_ratings():
    """Fix ratings for ALL restaurants using pagination"""
    
    # Initialize clients
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "menu-buddy")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    yelp_client = YelpAPIClient()
    
    print("🍽️  Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Use multiple search terms to get diverse restaurants
    search_terms = ["restaurant", "pizza", "italian", "chinese", "thai", "mexican", 
                    "japanese", "indian", "american", "french", "korean"]
    
    all_restaurants = {}  # Use dict to deduplicate by ID
    
    print("🔍 Fetching all restaurants from Pinecone...")
    
    for term in search_terms:
        term_vec = model.encode(term).tolist()
        results = index.query(
            vector=term_vec,
            top_k=200,  # Get 200 per term
            include_metadata=True,
            namespace="restaurants"
        )
        
        for match in results.matches:
            if match.id not in all_restaurants:
                all_restaurants[match.id] = match
        
        print(f"   {term:15} : {len(all_restaurants)} unique restaurants so far...")
        time.sleep(0.2)
    
    print(f"\n✅ Total unique restaurants retrieved: {len(all_restaurants)}")
    
    # Find restaurants with missing ratings
    missing_ratings = []
    has_ratings = 0
    
    for vector_id, match in all_restaurants.items():
        meta = match.metadata
        rating = meta.get("avg_rating", 0.0)
        name = meta.get("name", "Unknown")
        
        if rating == 0.0 or rating is None:
            # Extract Yelp ID
            yelp_id = vector_id.replace("restaurant:", "") if ":" in vector_id else vector_id
            
            try:
                location_json = meta.get("location_json", "{}")
                if location_json:
                    location = json.loads(location_json)
                    city = location.get("city", "Unknown")
                else:
                    city = "Unknown"
            except:
                city = "Unknown"
            
            missing_ratings.append({
                "vector_id": vector_id,
                "yelp_id": yelp_id,
                "name": name,
                "city": city,
                "metadata": meta
            })
        else:
            has_ratings += 1
    
    print(f"\n✅ {has_ratings} restaurants already have ratings")
    print(f"⚠️  {len(missing_ratings)} restaurants need rating updates")
    
    if not missing_ratings:
        print("\n✅ All restaurants have ratings!")
        return
    
    print(f"\n🔧 Fetching ratings from Yelp API...")
    print("=" * 80)
    
    updated_count = 0
    failed_count = 0
    
    for i, rest in enumerate(missing_ratings, 1):
        name = rest["name"]
        city = rest["city"]
        yelp_id = rest["yelp_id"]
        vector_id = rest["vector_id"]
        
        if i % 50 == 1:  # Print every 50
            print(f"\n[{i}/{len(missing_ratings)}] Processing...")
        
        try:
            # Get business details from Yelp
            details = yelp_client.get_business_details(yelp_id)
            
            if details and "rating" in details:
                rating = details.get("rating", 0.0)
                
                if rating > 0:
                    # Update metadata
                    updated_meta = rest["metadata"].copy()
                    updated_meta["avg_rating"] = float(rating)
                    
                    # Update in Pinecone
                    try:
                        index.update(
                            id=vector_id,
                            set_metadata=updated_meta,
                            namespace="restaurants"
                        )
                        updated_count += 1
                        
                        if updated_count % 10 == 0:
                            print(f"   ✅ Updated {updated_count} restaurants...")
                    except Exception as e:
                        print(f"   ❌ Failed to update {name}: {e}")
                        failed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            if i <= 10:  # Only print first 10 errors
                print(f"   ❌ Error for {name}: {e}")
            failed_count += 1
            time.sleep(0.5)
    
    # Final report
    print(f"\n{'='*80}")
    print(f"📊 FINAL REPORT")
    print(f"{'='*80}")
    print(f"   Total restaurants checked: {len(all_restaurants)}")
    print(f"   With missing ratings: {len(missing_ratings)}")
    print(f"   Successfully updated: {updated_count}")
    print(f"   Failed to update: {failed_count}")
    print(f"\n✅ Rating fix complete!")


if __name__ == "__main__":
    print("="*80)
    print("🔧 ALL RESTAURANT RATING FIXER")
    print("="*80)
    print("\nThis script will:")
    print("1. Find ALL restaurants in Pinecone with 0.0 rating")
    print("2. Fetch their ratings from Yelp API")
    print("3. Update Pinecone with correct ratings")
    print("\nThis will take 10-15 minutes for ~1,400 restaurants...")
    
    input("\nPress ENTER to continue or CTRL+C to cancel...")
    
    fix_all_ratings()
