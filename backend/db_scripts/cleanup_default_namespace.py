"""Delete vectors from default namespace and keep only restaurants namespace"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "menu-buddy"

print("🗑️  Cleaning up default namespace...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Get stats before
stats_before = index.describe_index_stats()
print(f"\n📊 Before cleanup:")
print(f"   Total vectors: {stats_before.get('total_vector_count', 0)}")
print(f"   Namespaces: {stats_before.get('namespaces', {})}")

# Delete all vectors from default namespace
print(f"\n🗑️  Deleting all vectors from '__default__' namespace...")
index.delete(delete_all=True, namespace="")  # Empty string = default namespace

# Get stats after
import time
time.sleep(2)  # Wait for deletion to propagate

stats_after = index.describe_index_stats()
print(f"\n✅ After cleanup:")
print(f"   Total vectors: {stats_after.get('total_vector_count', 0)}")
print(f"   Namespaces: {stats_after.get('namespaces', {})}")

print(f"\n✨ Done! Now re-run your populate commands to upload to the correct namespace.")
