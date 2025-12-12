import { createContext, useState, useContext, useEffect } from 'react'
import axios from 'axios'
import { API_URL } from '../config'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(localStorage.getItem('token'))
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      // Verify token and get user info
      axios.get(`${API_URL}/api/auth/me`)
        .then(response => {
          setUser(response.data)
        })
        .catch(() => {
          // Token invalid, clear it
          localStorage.removeItem('token')
          setToken(null)
          delete axios.defaults.headers.common['Authorization']
        })
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [token])

  const login = async (username, password) => {
    const formData = new URLSearchParams()
    formData.append('username', username)
    formData.append('password', password)
    
    const response = await axios.post(`${API_URL}/api/auth/login`, formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
    
    const { access_token, user_id, username: userUsername } = response.data
    localStorage.setItem('token', access_token)
    setToken(access_token)
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
    
    // Get full user info
    const userResponse = await axios.get(`${API_URL}/api/auth/me`)
    setUser(userResponse.data)
    
    return response.data
  }

  const signup = async (email, username, password, dietType = 'mix') => {
    try {
      const response = await axios.post(`${API_URL}/api/auth/signup`, {
        email,
        username,
        password,
        diet_type: dietType
      })
      
      const { access_token } = response.data
      localStorage.setItem('token', access_token)
      localStorage.setItem('swaad_diet_type', dietType)
      setToken(access_token)
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
      
      // Get full user info
      const userResponse = await axios.get(`${API_URL}/api/auth/me`)
      setUser(userResponse.data)
      
      return response.data
    } catch (error) {
      console.error('Signup error:', error)
      throw error
    }
  }

  const googleLogin = async (userInfo) => {
    try {
      // Send user info directly (simpler approach)
      const response = await axios.post(`${API_URL}/api/auth/google`, {
        email: userInfo.email,
        name: userInfo.name,
        picture: userInfo.picture,
        google_id: userInfo.id
      })
      
      const { access_token } = response.data
      localStorage.setItem('token', access_token)
      setToken(access_token)
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
      
      // Get full user info
      const userResponse = await axios.get(`${API_URL}/api/auth/me`)
      setUser(userResponse.data)
      
      return response.data
    } catch (error) {
      console.error('Google login error:', error)
      throw error
    }
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
    delete axios.defaults.headers.common['Authorization']
  }

  const value = {
    user,
    token,
    login,
    signup,
    googleLogin,
    logout,
    loading,
    isAuthenticated: !!token
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

