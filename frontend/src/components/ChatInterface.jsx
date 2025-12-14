import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import RestaurantCard from './RestaurantCard'
import './ChatInterface.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function ChatInterface() {
  const [selectedUser, setSelectedUser] = useState('default')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hi! I\'m your food recommendation assistant. Tell me what you\'re craving, and I\'ll suggest the perfect restaurants and dishes for you! 🍽️',
      type: 'text'
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const users = [
    { id: 'default', name: 'Default User (Veg, Nuts Allergy)' },
    { id: 'dummy2', name: 'User 2 (Non-Veg, Shellfish Allergy)' },
    { id: 'dummy3', name: 'User 3 (Mix, Gluten Allergy)' }
  ]

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage, type: 'text' }])
    setLoading(true)

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        query: userMessage,
        user_key: selectedUser
      })

      const data = response.data

      // Add text response
      const textResponse = data.response?.text || 'Sorry, I couldn\'t process that request.'
      setMessages(prev => [...prev, { role: 'assistant', content: textResponse, type: 'text' }])

      // Add restaurant cards if available
      if (data.menu_buddy?.recommendations && data.menu_buddy.recommendations.length > 0) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.menu_buddy.recommendations,
          type: 'restaurants'
        }])
      }

    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Sorry, there was an error processing your request. Please make sure the backend server is running on port 8000.',
        type: 'text'
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-container">
      <div className="user-selector">
        <label htmlFor="user-select">Current User: </label>
        <select 
          id="user-select" 
          value={selectedUser} 
          onChange={(e) => {
            setSelectedUser(e.target.value)
            setMessages([{
              role: 'assistant',
              content: `Switched to ${users.find(u => u.id === e.target.value).name}. How can I help you today?`,
              type: 'text'
            }])
          }}
        >
          {users.map(user => (
            <option key={user.id} value={user.id}>
              {user.name}
            </option>
          ))}
        </select>
      </div>
      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-avatar">
              {message.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
              {message.type === 'text' ? (
                message.content.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))
              ) : message.type === 'restaurants' ? (
                <div className="restaurants-grid">
                  {message.content.map((restaurant, idx) => (
                    <RestaurantCard key={idx} restaurant={restaurant} />
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-container" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="What are you craving? (e.g., 'I want spicy Thai food')"
          disabled={loading}
          className="chat-input"
        />
        <button type="submit" disabled={loading || !input.trim()} className="send-button">
          {loading ? '⏳' : '🚀'}
        </button>
      </form>
    </div>
  )
}

export default ChatInterface

