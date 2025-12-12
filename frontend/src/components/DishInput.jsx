import { useState } from 'react'
import axios from 'axios'
import { API_URL } from '../config'
import './DishInput.css'

function DishInput({ onProfileCreated, initialDishes = [], onDishesUpdate }) {
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!prompt.trim()) {
      setError('Please describe your taste preferences')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await axios.post(`${API_URL}/api/create-profile-ai`, {
        prompt: prompt
      })
      
      // The response is the UserProfile object which now includes favorite_dishes
      const profile = response.data
      const dishes = profile.favorite_dishes || []
      
      // Pass both profile and dishes to the callback
      onProfileCreated(profile, dishes)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create profile. Please try again.')
      setLoading(false)
    }
  }

  return (
    <div className="dish-input">
      <h2>Tell us about your taste!</h2>
      <p className="subtitle">Describe your favorite appetizers, mains, desserts, and any allergies.</p>
      
      <form onSubmit={handleSubmit} className="preference-form">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. I love spicy food, especially Chicken Tikka Masala. For dessert I like chocolate cake. I am allergic to peanuts."
          rows={6}
          className="preference-input"
          disabled={loading}
        />
        
        {error && <div className="error-message">{error}</div>}
        
        <button type="submit" className="create-profile-button" disabled={loading || !prompt.trim()}>
          {loading ? 'Analyzing...' : 'Create Flavor Profile'}
        </button>
      </form>
    </div>
  )
}

export default DishInput

