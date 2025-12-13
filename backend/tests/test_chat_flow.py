import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/chat"

def print_response(response):
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

def test_chat():
    print("🧪 Testing /api/chat endpoint...\n")

    # Test 1: Simple query with location
    print("Test 1: Simple query for spicy Indian food in San Francisco")
    print("-----------------------------------------------------------")
    payload = {
        "query": "I want spicy Indian food",
        "location": "San Francisco, CA",
        "diet_type": "mix"
    }
    try:
        r = requests.post(BASE_URL, json=payload)
        print_response(r)
    except Exception as e:
        print(f"Request failed: {e}")

    print("\n\n")

    # Test 2: Query without location (should ask for location)
    print("Test 2: Query without location (should ask for location)")
    print("--------------------------------------------------------")
    payload = {
        "query": "I want pizza",
        "diet_type": "mix"
    }
    try:
        r = requests.post(BASE_URL, json=payload)
        print_response(r)
    except Exception as e:
        print(f"Request failed: {e}")

    print("\n\n")

    # Test 3: Vegetarian query
    print("Test 3: Vegetarian query in New York")
    print("------------------------------------")
    payload = {
        "query": "I want a veg burger",
        "location": "New York, NY",
        "diet_type": "mix" # Should be overridden by query
    }
    try:
        r = requests.post(BASE_URL, json=payload)
        print_response(r)
    except Exception as e:
        print(f"Request failed: {e}")

    print("\n\n")
    
    # Test 4: Specific Dish at Restaurant
    print("Test 4: Specific Dish (Pasta at Don Angie)")
    print("------------------------------------")
    payload = {
        "query": "I want pasta at Don Angie",
        "location": "New York, NY",
        "diet_type": "mix"
    }
    try:
        r = requests.post(BASE_URL, json=payload)
        print_response(r)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_chat()
