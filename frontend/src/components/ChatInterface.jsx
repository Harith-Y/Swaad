import React, { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';
import { API_URL } from '../config';

const ChatInterface = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { text: "Hi! I can help you find restaurants using Yelp. What are you looking for?", sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatId, setChatId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleNewChat = () => {
    setChatId(null);
    setMessages([
      { text: "Hi! I can help you find restaurants using Yelp. What are you looking for?", sender: 'bot' }
    ]);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userMessage = inputValue.trim();
    let storedProfile = null;
    let storedDishes = null;
    try {
      const rawProfile = localStorage.getItem('swaad_flavor_profile');
      if (rawProfile) storedProfile = JSON.parse(rawProfile);
    } catch (err) {
      storedProfile = null;
    }
    try {
      const rawDishes = localStorage.getItem('swaad_favorite_dishes');
      if (rawDishes) storedDishes = JSON.parse(rawDishes);
    } catch (err) {
      storedDishes = null;
    }

    setMessages(prev => [...prev, { text: userMessage, sender: 'user' }]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          chat_id: chatId,
          user_profile: storedProfile,
          favorite_dishes: storedDishes
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      // Update chat ID for next request
      if (data.chat_id) {
        setChatId(data.chat_id);
      }

      // Extract text from Yelp response
      const botResponse = data.response?.text || "Sorry, I couldn't understand that.";
      
      // Extract businesses if available
      let businesses = [];
      if (data.menu_buddy && data.menu_buddy.recommendations && data.menu_buddy.recommendations.length > 0) {
        businesses = data.menu_buddy.recommendations.map((r) => ({
          id: r.id,
          name: r.name,
          url: r.url,
          rating: r.avg_rating,
          location: r.location
        }));
      } else if (data.entities && data.entities.length > 0 && data.entities[0].businesses) {
        businesses = data.entities[0].businesses;
      }

      setMessages(prev => [...prev, { text: botResponse, sender: 'bot', businesses }]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, { text: "Sorry, something went wrong. Please try again.", sender: 'bot' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-widget">
      {isOpen && (
        <div className="chat-window">
          <div className="chat-header">
            <h3>Swaad Assistant</h3>
            <div className="header-actions">
              <button className="new-chat-btn" onClick={handleNewChat} title="Start New Chat">
                +
              </button>
              <button className="close-btn" onClick={toggleChat}>×</button>
            </div>
          </div>
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message-container ${msg.sender}`}>
                <div className={`message ${msg.sender}`}>
                  {msg.text}
                </div>
                {msg.businesses && msg.businesses.length > 0 && (
                  <div className="business-cards">
                    {msg.businesses.map((business) => (
                      <a 
                        key={business.id} 
                        href={business.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="business-card"
                      >
                        <div className="business-info">
                          <h4>{business.name}</h4>
                          {business.rating && <span className="rating">★ {business.rating}</span>}
                        </div>
                        {business.location && (
                          <p className="business-address">
                            {business.location.address1}, {business.location.city}
                          </p>
                        )}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {isLoading && <div className="typing-indicator">Typing...</div>}
            <div ref={messagesEndRef} />
          </div>
          <form className="chat-input-area" onSubmit={handleSendMessage}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about restaurants..."
              disabled={isLoading}
            />
            <button type="submit" className="send-btn" disabled={isLoading || !inputValue.trim()}>
              ➤
            </button>
          </form>
        </div>
      )}
      <button className="chat-toggle-btn" onClick={toggleChat}>
        💬
      </button>
    </div>
  );
};

export default ChatInterface;
