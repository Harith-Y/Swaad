import { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import './App.css'

function App() {
  return (
    <div className="App">
      <header className="app-header">
        <h1>🍽️ Swaad</h1>
        <p>AI-Powered Restaurant Recommendations</p>
      </header>
      <main className="app-main">
        <ChatInterface />
      </main>
    </div>
  )
}

export default App

