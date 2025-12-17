"""
Fix "Unknown" city restaurants by extracting city from location_json
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def main():
    """Fix Unknown city restaurants"""
    
    print("Fetching restaurants with Unknown city...\n")
    
    pc_index = get_pinecone_index()
    
    # Query for all restaurants
    dummy_vector = [0.0] * 384
    results = pc_index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace="restaurants"
    )
    
    # Find Unknown city restaurants
    unknown_restaurants = []
    
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        city = meta.get("city", "")
        
        if not city or city == "Unknown" or city.strip() == "":
            unknown_restaurants.append({
                "id": match.get("id"),
                "name": meta.get("name"),
                "metadata": meta
            })
    
    print(f"Found {len(unknown_restaurants)} restaurants with Unknown city\n")
    
    if not unknown_restaurants:
        print("✅ All restaurants have city set!")
        return
    
    # Try to extract city from location_json
    updates = []
    city_extracted = {}
    
    for rest in unknown_restaurants:
        meta = rest['metadata']
        location_json = meta.get('location_json', '')
        
        city = None
        
        # Try to parse location_json
        if location_json:
            try:
                location = json.loads(location_json)
                city = location.get('city', '')
                
                if city and city.strip():
                    city = city.strip()
                    city_extracted[city] = city_extracted.get(city, 0) + 1
                    
                    # Update metadata
                    meta['city'] = city
                    updates.append({
                        "id": rest['id'],
                        "metadata": meta
                    })
            except:
                pass
    
    print(f"Successfully extracted city for {len(updates)} restaurants\n")
    
    if city_extracted:
        print("Cities found:")
        for city, count in sorted(city_extracted.items()):
            print(f"  {city}: {count}")
        print()
    
    remaining = len(unknown_restaurants) - len(updates)
    if remaining > 0:
        print(f"⚠️  {remaining} restaurants still have no city (no location_json or invalid)\n")
    
    if updates:
        confirm = input(f"Update {len(updates)} restaurants with extracted cities? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\nUpdating Pinecone...")
            
            # Update in batches
            batch_size = 100
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]
                
                # Fetch vectors for each update
                upsert_data = []
                batch_ids = [update['id'] for update in batch]
                result = pc_index.fetch(ids=batch_ids, namespace="restaurants")
                
                for update in batch:
                    if result.get('vectors') and update['id'] in result['vectors']:
                        vector_data = result['vectors'][update['id']]
                        upsert_data.append({
                            "id": update['id'],
                            "values": vector_data.get('values'),
                            "metadata": update['metadata']
                        })
                
                if upsert_data:
                    pc_index.upsert(vectors=upsert_data, namespace="restaurants")
                    print(f"  Updated batch {i // batch_size + 1}/{(len(updates) + batch_size - 1) // batch_size}")
            
            print(f"\n✅ Successfully updated {len(updates)} restaurants with city field!")
            
            # Show updated stats
            print(f"\nUpdated cities:")
            for city, count in sorted(city_extracted.items()):
                print(f"  {city}: {count} restaurants")
        else:
            print("Aborted.")


if __name__ == "__main__":
    main()
