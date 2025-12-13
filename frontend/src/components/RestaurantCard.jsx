import './RestaurantCard.css'

function RestaurantCard({ restaurant }) {
  return (
    <div className="restaurant-card">
      <div className="restaurant-header">
        <h3 className="restaurant-name">{restaurant.name}</h3>
        <div className="restaurant-rating">
          <span className="rating-star">⭐</span>
          <span className="rating-value">{restaurant.rating}/5</span>
        </div>
      </div>
      
      <div className="restaurant-cuisine">
        <span className="cuisine-icon">🍴</span>
        <span className="cuisine-text">{restaurant.cuisine_types.join(', ')}</span>
      </div>

      {restaurant.recommended_dishes && restaurant.recommended_dishes.length > 0 && (
        <div className="dishes-section">
          <h4 className="dishes-title">🍽️ Recommended Dishes</h4>
          <ul className="dishes-list">
            {restaurant.recommended_dishes.slice(0, 5).map((dish, idx) => (
              <li key={idx} className="dish-item">
                <span className="dish-number">{idx + 1}.</span>
                <span className="dish-name">{dish.name}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default RestaurantCard

