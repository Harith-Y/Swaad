"""
Populate Chicago Restaurants Script

Quick script to populate Chicago restaurants using the master populator.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from master_populate import RestaurantPopulator


def main():
    """Populate Chicago restaurants"""
    
    print("=" * 80)
    print("🏙️  CHICAGO RESTAURANT POPULATION")
    print("=" * 80)
    print("\nThis script will search and populate Chicago restaurants")
    print("with comprehensive dish data using multiple search strategies.\n")
    
    # Chicago locations
    locations = [
        "Chicago, IL",
        "Downtown Chicago, IL",
    ]
    
    print(f"Target locations: {', '.join(locations)}")
    print(f"Search strategies: 15+ cuisine types")
    print(f"Limit per category: 50 restaurants\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return
    
    # Create populator and run
    populator = RestaurantPopulator()
    populator.populate_locations(locations, limit_per_category=50)
    
    print("\n✅ Chicago population complete!")
    print("The backend will automatically pick up these restaurants on next query.")


if __name__ == "__main__":
    main()
