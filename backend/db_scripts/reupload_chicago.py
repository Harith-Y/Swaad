"""
Re-upload Chicago restaurants with fixed city metadata
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from restaurant_to_pinecone import process_restaurants, upload_to_pinecone


def main():
    """Re-upload Chicago restaurants"""
    
    json_file = "master_restaurants_20251217_033916.json"
    
    print(f"Loading restaurants from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} restaurants\n")
    
    # Filter for Chicago only
    chicago = []
    for r in data:
        location = r.get('location', {})
        if isinstance(location, dict):
            city = location.get('city', '')
            if isinstance(city, str) and 'chicago' in city.lower():
                chicago.append(r)
    
    print(f"Found {len(chicago)} Chicago restaurants to re-upload\n")
    
    # Process all at once
    print("Processing restaurants...")
    vectors = process_restaurants(chicago)
    
    print(f"\nUploading {len(vectors)} vectors to Pinecone...")
    upload_to_pinecone(vectors)
    
    print(f"\n✅ Re-upload complete!")
    print(f"   Total restaurants: {len(chicago)}")
    print(f"   Vectors created: {len(vectors)}")


if __name__ == "__main__":
    main()
