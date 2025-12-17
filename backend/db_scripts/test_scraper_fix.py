"""
Test the fixed menu scraper on Au Cheval
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from menu_url_scraper import MenuURLScraper


def main():
    scraper = MenuURLScraper()
    
    # Au Cheval menu URL (you'll need to get this from the database)
    # For now, let's test with a sample
    
    print("Testing menu scraper fixes...\n")
    
    # Test cleaning function
    test_dishes = [
        "single cheeseburger *•",
        "thick cut peppered bacon •",
        "with eggs",
        "~add chicken breast - 8.99| add avocado 2.99~",
        "General Jane's Honey-Fried Chicken",
        "add extra cheese $2.50"
    ]
    
    print("Test Dish Cleaning:")
    print("="*60)
    for dish in test_dishes:
        cleaned = scraper._clean_dish_name(dish)
        valid = scraper._is_valid_dish_name(cleaned)
        print(f"Original: {dish!r}")
        print(f"Cleaned:  {cleaned!r}")
        print(f"Valid:    {valid}")
        print()
    
    print("\nExpected results:")
    print("✅ 'Single Cheeseburger' - valid")
    print("✅ 'Thick Cut Peppered Bacon' - valid")
    print("❌ 'eggs' - invalid (too short, starts with 'with')")
    print("❌ '' - invalid (cleaned to empty)")
    print("✅ 'General Jane's Honey-Fried Chicken' - valid")
    print("❌ 'extra cheese' - invalid (starts with 'add')")


if __name__ == "__main__":
    main()
