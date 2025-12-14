from dish_processing import check_location_match

def test_location_match():
    print("Testing check_location_match...")
    
    # Test 1: Exact match
    assert check_location_match("New York", "New York") == True
    print("✓ Exact match passed")
    
    # Test 2: Containment
    assert check_location_match("Manhattan", "New York City") == True
    print("✓ Containment passed")
    
    # Test 3: Mismatch
    assert check_location_match("San Francisco", "New York") == False
    print("✓ Mismatch passed")
    
    # Test 4: Case insensitivity
    assert check_location_match("san francisco", "San Francisco") == True
    print("✓ Case insensitivity passed")

    print("All tests passed!")

if __name__ == "__main__":
    test_location_match()