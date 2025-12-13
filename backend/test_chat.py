import requests
import json

url = "http://localhost:8000/api/chat"

payload = {
    "query": "I want some spicy indian food",
    "user_key": "default",
    "location": "New York, NY"
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    print("Response Status:", response.status_code)
    print("\nAI Response Text:")
    print(data.get("response", {}).get("text"))
    
    print("\nRecommendations:")
    recommendations = data.get("menu_buddy", {}).get("recommendations", [])
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec.get('name')} (Score: {rec.get('score'):.2f})")
        print(f"   Cuisine: {rec.get('cuisine_types')}")
        print(f"   Recommended Dishes: {[d.get('name') for d in rec.get('recommended_dishes', [])]}")
        print("-" * 30)
        
except Exception as e:
    print(f"Error: {e}")
    if 'response' in locals():
        print(response.text)
