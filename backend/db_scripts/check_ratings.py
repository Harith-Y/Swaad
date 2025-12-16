"""
Quick script to check rating values in Pinecone
"""
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment
load_dotenv()

def check_ratings():
    """Check rating distribution in Pinecone"""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "menu-buddy")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Query for a sample of restaurants
    results = index.query(
        vector=[0.0] * 384,
        top_k=100,
        include_metadata=True,
        namespace="restaurants"
    )
    
    print(f"🔍 Checking ratings for {len(results.matches)} restaurants...\n")
    
    rating_counts = {
        "0.0": 0,
        "1.0-2.0": 0,
        "2.0-3.0": 0,
        "3.0-4.0": 0,
        "4.0-5.0": 0
    }
    
    sample_restaurants = []
    
    for match in results.matches:
        meta = match.metadata
        rating = meta.get("avg_rating", 0.0)
        name = meta.get("name", "Unknown")
        city = "Unknown"
        
        try:
            import json
            location_json = meta.get("location_json", "{}")
            if location_json:
                location = json.loads(location_json)
                city = location.get("city", "Unknown")
        except:
            pass
        
        # Categorize rating
        if rating == 0.0:
            rating_counts["0.0"] += 1
        elif rating < 2.0:
            rating_counts["1.0-2.0"] += 1
        elif rating < 3.0:
            rating_counts["2.0-3.0"] += 1
        elif rating < 4.0:
            rating_counts["3.0-4.0"] += 1
        else:
            rating_counts["4.0-5.0"] += 1
        
        # Save first 10 for display
        if len(sample_restaurants) < 10:
            sample_restaurants.append({
                "name": name,
                "city": city,
                "rating": rating
            })
    
    # Print distribution
    print("📊 Rating Distribution:")
    print("=" * 60)
    total = len(results.matches)
    for range_name, count in rating_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"   {range_name:12} : {count:4} restaurants ({percentage:5.1f}%)")
    
    print(f"\n📝 Sample Restaurants:")
    print("=" * 60)
    for rest in sample_restaurants:
        print(f"   {rest['name']:40} | {rest['city']:15} | ⭐ {rest['rating']}/5")
    
    # Check if ratings are actually 0
    zero_rating_count = rating_counts["0.0"]
    if zero_rating_count > total * 0.5:
        print(f"\n⚠️  WARNING: {zero_rating_count}/{total} ({zero_rating_count/total*100:.1f}%) restaurants have 0.0 rating!")
        print("   This suggests the Yelp API may not be returning ratings, or they weren't saved properly.")
    else:
        print(f"\n✅ Ratings look good! Only {zero_rating_count}/{total} have 0.0 rating.")


if __name__ == "__main__":
    check_ratings()
