import { useState, useEffect } from 'react'
import DishInput from './components/DishInput'
import FlavorProfile from './components/FlavorProfile'
import MenuUpload from './components/MenuUpload'
import Recommendations from './components/Recommendations'
import Profile from './components/Profile'
import ChatInterface from './components/ChatInterface'
import './App.css'
import { API_URL } from './config'

function App() {
  const isChatRoute = window.location.pathname === '/chat' || window.location.pathname === '/chat/'
  const [currentView, setCurrentView] = useState('main') // 'main' or 'profile'
  const [step, setStep] = useState(1)
  const [userProfile, setUserProfile] = useState(null)
  const [menuDishes, setMenuDishes] = useState([])
  const [recommendations, setRecommendations] = useState(null)
  const [favoriteDishes, setFavoriteDishes] = useState([])

  if (isChatRoute) {
    return (
      <div className="app">
        <ChatInterface fullScreen />
      </div>
    )
  }

  // Load saved profile from localStorage
  useEffect(() => {
    const loadDummyUser = async () => {
      try {
        const resp = await fetch(`${API_URL}/api/dummy-user`)
        if (!resp.ok) return
        const data = await resp.json()
        if (data && data.flavor_profile) {
          setUserProfile(data.flavor_profile)
          localStorage.setItem('swaad_flavor_profile', JSON.stringify(data.flavor_profile))
        }
        if (data && data.favorite_dishes) {
          setFavoriteDishes(data.favorite_dishes)
          localStorage.setItem('swaad_favorite_dishes', JSON.stringify(data.favorite_dishes))
        }
        if (data && data.diet_type) {
          localStorage.setItem('swaad_diet_type', data.diet_type)
        }
        if (data && data.location) {
          localStorage.setItem('swaad_location', data.location)
        }
      } catch (e) {
        // ignore
      }
    }
    loadDummyUser()

    const savedProfile = localStorage.getItem('swaad_flavor_profile')
    const savedDishes = localStorage.getItem('swaad_favorite_dishes')
    if (savedProfile) {
      try {
        setUserProfile(JSON.parse(savedProfile))
      } catch (e) {
        console.error('Error loading saved profile:', e)
      }
    }
    if (savedDishes) {
      try {
        setFavoriteDishes(JSON.parse(savedDishes))
      } catch (e) {
        console.error('Error loading saved dishes:', e)
      }
    }
  }, [])

  const handleProfileCreated = (profile, dishes = []) => {
    setUserProfile(profile)
    setFavoriteDishes(dishes)
    // Save to localStorage
    localStorage.setItem('swaad_flavor_profile', JSON.stringify(profile))
    localStorage.setItem('swaad_favorite_dishes', JSON.stringify(dishes))

    // Persist to backend dummy user map
    try {
      fetch(`${API_URL}/api/dummy-user`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          flavor_profile: profile,
          favorite_dishes: dishes,
          diet_type: localStorage.getItem('swaad_diet_type')
        })
      })
    } catch (e) {
      // ignore
    }

    setStep(2)
  }

  const handleMenuProcessed = (dishes) => {
    setMenuDishes(dishes)
    setStep(3)
  }

  const handleRecommendationsReceived = (recs) => {
    setRecommendations(recs)
  }

  const resetFlow = () => {
    setStep(1)
    setUserProfile(null)
    setMenuDishes([])
    setRecommendations(null)
  }

  if (currentView === 'profile') {
    return (
      <div className="app">
        <ChatInterface />
        <div className="container">
          <nav className="app-nav">
            <h1 className="nav-title">🍽️ Swaad</h1>
            <div className="nav-buttons">
              <button 
                className="nav-button"
                onClick={() => setCurrentView('main')}
              >
                ← Back to Main
              </button>
            </div>
          </nav>
          <Profile 
            userProfile={userProfile}
            favoriteDishes={favoriteDishes}
            onProfileUpdate={handleProfileCreated}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <ChatInterface />
      <div className="container">
        <nav className="app-nav">
          <h1 className="nav-title">🍽️ Swaad</h1>
          <div className="nav-buttons">
            <button 
              className="nav-button"
              onClick={() => setCurrentView('profile')}
            >
              My Profile
            </button>
          </div>
        </nav>
        
        <header className="header">
          <h1>🍽️ Swaad</h1>
          <p>Discover dishes that match your flavor profile</p>
        </header>

        <div className="steps-indicator">
          <div className={`step ${step >= 1 ? 'active' : ''}`}>
            <div className="step-number">1</div>
            <div className="step-label">Your Preferences</div>
          </div>
          <div className={`step ${step >= 2 ? 'active' : ''}`}>
            <div className="step-number">2</div>
            <div className="step-label">Menu Upload</div>
          </div>
          <div className={`step ${step >= 3 ? 'active' : ''}`}>
            <div className="step-number">3</div>
            <div className="step-label">Recommendations</div>
          </div>
        </div>

        <main className="main-content">
          {step === 1 && (
            <div className="step-content">
              <DishInput 
                onProfileCreated={handleProfileCreated}
                initialDishes={favoriteDishes}
              />
            </div>
          )}

          {step === 2 && userProfile && (
            <div className="step-content">
              <FlavorProfile profile={userProfile} />
              <MenuUpload 
                onMenuProcessed={handleMenuProcessed}
                userProfile={userProfile}
                onRecommendationsReceived={handleRecommendationsReceived}
              />
            </div>
          )}

          {step === 3 && recommendations && (
            <div className="step-content">
              <Recommendations 
                recommendations={recommendations}
                menuDishes={menuDishes}
              />
              <button className="reset-button" onClick={resetFlow}>
                Start Over
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App

