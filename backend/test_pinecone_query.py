"""
Test script to check Pinecone index and query restaurants.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "menu-buddy")

print(f"Connecting to Pinecone...")
print(f"API Key: {PINECONE_API_KEY[:20]}...")
print(f"Index: {PINECONE_INDEX}")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Get index stats
print("\n=== Index Stats ===")
stats = index.describe_index_stats()
print(f"Total vectors: {stats.get('total_vector_count', 0)}")
print(f"Dimension: {stats.get('dimension', 0)}")
print(f"Namespaces: {stats.get('namespaces', {})}")

# Try to fetch some vectors
print("\n=== Fetching sample vectors ===")
try:
    # Try to fetch by ID
    fetch_result = index.fetch(ids=["restaurant:1", "restaurant:2", "restaurant:3"])
    print(f"Fetched {len(fetch_result.get('vectors', {}))} vectors")
    for vec_id, vec_data in fetch_result.get('vectors', {}).items():
        meta = vec_data.get('metadata', {})
        print(f"  - {vec_id}: {meta.get('name', 'N/A')}")
except Exception as e:
    print(f"Error fetching: {e}")

# Try a simple query
print("\n=== Testing query (default namespace) ===")
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_text = "Indian restaurant"
    query_vec = model.encode(query_text).tolist()

    print(f"Query: '{query_text}'")
    print(f"Vector dimension: {len(query_vec)}")

    results = index.query(vector=query_vec, top_k=10, include_metadata=True)
    matches = results.get('matches', [])

    print(f"Found {len(matches)} matches in default namespace")

except Exception as e:
    print(f"Error querying: {e}")
    import traceback
    traceback.print_exc()

# Try querying the restaurants namespace
print("\n=== Testing query (restaurants namespace) ===")
try:
    results = index.query(vector=query_vec, top_k=10, include_metadata=True, namespace="restaurants")
    matches = results.get('matches', [])

    print(f"Found {len(matches)} matches in 'restaurants' namespace:")
    for i, match in enumerate(matches, 1):
        meta = match.get('metadata', {})
        score = match.get('score', 0)
        print(f"  {i}. {meta.get('name', 'N/A')} (score: {score:.4f})")
        print(f"     Cuisine: {meta.get('cuisine_types', [])}")
        print(f"     Menu items: {meta.get('menu_items', [])[:3]}")

except Exception as e:
    print(f"Error querying: {e}")
    import traceback
    traceback.print_exc()

