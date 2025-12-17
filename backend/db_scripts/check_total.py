"""
Check total restaurant count in Pinecone
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def check_total():
    """Check total restaurant count"""
    try:
        pc_index = get_pinecone_index()
        
        # Get index stats
        stats = pc_index.describe_index_stats()
        
        print(f"📊 Pinecone Index Statistics:\n")
        print(f"Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"Namespaces: {stats.get('namespaces', {})}")
        
        # Query for sample restaurants
        dummy_vector = [0.0] * 384
        results = pc_index.query(
            vector=dummy_vector,
            top_k=10,
            include_metadata=True,
            namespace="restaurants"
        )
        
        print(f"\n🍽️  Sample Restaurants (showing location format):")
        for i, match in enumerate(results.get("matches", [])[:10]):
            meta = match.get("metadata", {})
            location = meta.get("location", "")
            print(f"  {i+1}. {meta.get('name')}")
            print(f"     Location type: {type(location)}")
            print(f"     Location value: {repr(location)}")
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_total()
