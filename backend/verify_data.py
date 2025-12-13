import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def verify_pinecone_data():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX") or "menu-buddy"
    
    if not api_key:
        print("Error: PINECONE_API_KEY not found.")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    print(f"Connected to Pinecone index: {index_name}")
    
    # Check stats
    stats = index.describe_index_stats()
    print("\nIndex Stats:")
    print(json.dumps(stats.to_dict(), indent=2))
    
    # Check specifically for 'restaurants' namespace
    namespaces = stats.get('namespaces', {})
    if 'restaurants' in namespaces:
        count = namespaces['restaurants']['vector_count']
        print(f"\nFound {count} vectors in 'restaurants' namespace.")
        
        # Fetch a sample to see the data quality
        print("\nFetching a sample restaurant...")
        # We need a vector to query, or we can just fetch if we knew an ID. 
        # Since we don't know IDs, we'll do a dummy query.
        dummy_vec = [0.1] * 384 # Dimension depends on model, usually 384 for all-MiniLM-L6-v2
        
        try:
            results = index.query(
                vector=dummy_vec,
                top_k=3,
                namespace="restaurants",
                include_metadata=True
            )
            
            for match in results['matches']:
                md = match['metadata']
                print(f"\n--- Restaurant: {md.get('name')} ---")
                print(f"City: {md.get('city')}")
                print(f"Cuisine: {md.get('cuisine_types')}")
                print(f"Menu Items (first 5): {md.get('menu_items', [])[:5]}")
                print(f"Popular Dishes: {md.get('popular_dishes', [])[:5]}")
                print(f"Taste Vector: {[md.get(f'taste_{i}') for i in range(6)]}")
                
        except Exception as e:
            print(f"Error querying: {e}")
            
    else:
        print("\nWARNING: 'restaurants' namespace not found or empty!")

if __name__ == "__main__":
    verify_pinecone_data()
