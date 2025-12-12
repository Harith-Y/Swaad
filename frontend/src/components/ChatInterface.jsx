import React, { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';
import { API_URL } from '../config';

const ChatInterface = ({ fullScreen = false }) => {
  const [isOpen, setIsOpen] = useState(!!fullScreen);
  const [messages, setMessages] = useState([
    { text: "Hi! I can help you find restaurants using Yelp. What are you looking for?", sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatId, setChatId] = useState(null);
  const [userKey, setUserKey] = useState(() => {
    try {
      return localStorage.getItem('swaad_user_key') || 'default';
    } catch (e) {
      return 'default';
    }
  });
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (fullScreen) setIsOpen(true);
  }, [fullScreen]);

  useEffect(() => {
    try {
      localStorage.setItem('swaad_user_key', userKey);
    } catch (e) {
      // ignore
    }

    // Sync dummy-user metadata (diet_type/location) so chat calls stay consistent.
    const sync = async () => {
      try {
        const resp = await fetch(`${API_URL}/api/dummy-user?user_key=${encodeURIComponent(userKey)}`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && data.diet_type) {
          localStorage.setItem('swaad_diet_type', data.diet_type);
        }
        if (data && data.location) {
          localStorage.setItem('swaad_location', data.location);
        }
      } catch (e) {
        // ignore
      }
    };

    sync();
  }, [userKey]);

  const toggleChat = () => {
    if (fullScreen) return;
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
    let storedDietType = null;

    try {
      storedDietType = localStorage.getItem('swaad_diet_type');
    } catch (err) {
      storedDietType = null;
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
          diet_type: storedDietType,
          user_key: userKey
        }),
      });

      if (!response.ok) {
        let errBody = '';
        try {
          errBody = await response.text();
        } catch (e) {
          errBody = '';
        }
        console.error('Chat API error:', response.status, errBody);
        throw new Error(errBody || 'Failed to get response');
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
          location: r.location,
          recommended_dishes: r.recommended_dishes || []
        }));
      } else if (data.entities && data.entities.length > 0 && data.entities[0].businesses) {
        businesses = data.entities[0].businesses.map((b) => ({
          id: b.id,
          name: b.name,
          url: b.url,
          rating: b.rating,
          location: b.location,
          recommended_dishes: []
        }));
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
    <div className={`chat-widget ${fullScreen ? 'fullscreen' : ''}`}>
      {isOpen && (
        <div className={`chat-window ${fullScreen ? 'fullscreen' : ''}`}>
          <div className="chat-header">
            <h3>Swaad Assistant</h3>
            <div className="header-actions">
              <select
                className="user-select"
                value={userKey}
                onChange={(e) => {
                  setChatId(null);
                  setMessages([
                    { text: "Hi! I can help you find restaurants using Yelp. What are you looking for?", sender: 'bot' }
                  ]);
                  setUserKey(e.target.value);
                }}
                title="Select dummy user"
              >
                <option value="default">default</option>
                <option value="dummy2">dummy2</option>
                <option value="dummy3">dummy3</option>
              </select>
              <button className="new-chat-btn" onClick={handleNewChat} title="Start New Chat">
                +
              </button>
              {!fullScreen && <button className="close-btn" onClick={toggleChat}>×</button>}
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
                        {business.recommended_dishes && business.recommended_dishes.length > 0 && (
                          <div className="recommended-dishes">
                            <strong>Try:</strong> {business.recommended_dishes.slice(0, 3).map(d => typeof d === 'string' ? d : d.name).join(', ')}
                          </div>
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
      {!fullScreen && (
        <button className="chat-toggle-btn" onClick={toggleChat}>
          💬
        </button>
      )}
    </div>
  );
};

export default ChatInterface;
