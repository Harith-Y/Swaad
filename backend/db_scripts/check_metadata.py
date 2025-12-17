"""
Check Pinecone metadata structure
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pinecone_client import get_pinecone_index


def check_metadata():
    """Check metadata structure"""
    try:
        pc_index = get_pinecone_index()
        
        # Fetch a few records by ID
        dummy_vector = [0.0] * 384
        results = pc_index.query(
            vector=dummy_vector,
            top_k=20,
            include_metadata=True,
            namespace="restaurants"
        )
        
        print(f"Checking metadata structure...\n")
        
        for i, match in enumerate(results.get("matches", [])[:5]):
            meta = match.get("metadata", {})
            print(f"{i+1}. {meta.get('name')}")
            print(f"   Metadata keys: {list(meta.keys())}")
            
            # Check location field
            if 'location' in meta:
                print(f"   location type: {type(meta['location'])}")
                print(f"   location value: {meta['location']}")
            
            # Check city field
            if 'city' in meta:
                print(f"   city: {meta['city']}")
            
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_metadata()
