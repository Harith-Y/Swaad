from dish_processing import extract_location_from_query, extract_dish_and_restaurant

def test_extraction():
    print("Testing extraction logic...")
    
    # Test 1: Location Extraction
    query1 = "What about pasta for my friend in san fransisco"
    loc1 = extract_location_from_query(query1)
    print(f"Query: '{query1}' -> Location: '{loc1}'")
    assert loc1 is not None and "san" in loc1.lower()
    
    query2 = "I want to eat pasta"
    loc2 = extract_location_from_query(query2)
    print(f"Query: '{query2}' -> Location: '{loc2}'")
    assert loc2 is None
    
    # Test 2: Dish & Restaurant Extraction
    query3 = "I want the burger from Shake Shack"
    dr3 = extract_dish_and_restaurant(query3)
    print(f"Query: '{query3}' -> {dr3}")
    assert dr3 == ("burger", "shake shack")
    
    print("All extraction tests passed!")

if __name__ == "__main__":
    test_extraction()