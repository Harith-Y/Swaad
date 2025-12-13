#!/bin/bash

# Test script for the /api/chat endpoint

echo "🧪 Testing /api/chat endpoint..."
echo ""

# Check if server is running
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "❌ Server is not running on http://localhost:8000"
    echo "Please start the server first:"
    echo "  cd backend && python3.10 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    exit 1
fi

echo "✓ Server is running"
echo ""

# Test 1: Simple query with location
echo "Test 1: Simple query for spicy Indian food in San Francisco"
echo "-----------------------------------------------------------"
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want spicy Indian food",
    "location": "San Francisco, CA",
    "diet_type": "mix"
  }' | python3.10 -m json.tool

echo ""
echo ""

# Test 2: Query without location (should ask for location)
echo "Test 2: Query without location (should ask for location)"
echo "--------------------------------------------------------"
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want pizza",
    "diet_type": "mix"
  }' | python3.10 -m json.tool

echo ""
echo ""

# Test 3: Vegetarian query
echo "Test 3: Vegetarian query in New York"
echo "------------------------------------"
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want vegetarian food",
    "location": "New York, NY",
    "diet_type": "veg"
  }' | python3.10 -m json.tool

echo ""
echo "✅ Tests complete!"

