import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { chatAPI } from './services/api';

// Auth0 Configuration - Will use env vars or defaults
const AUTH0_DOMAIN = import.meta.env.VITE_AUTH0_DOMAIN || "dev-algorythm.us.auth0.com";
const AUTH0_CLIENT_ID = import.meta.env.VITE_AUTH0_CLIENT_ID || "your-client-id";
const AUTH0_REDIRECT_URI = window.location.origin;
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  sources?: any[];
  pdfUrl?: string;
}

const AlgoRythmAIApp: React.FC = () => {
  const [userName, setUserName] = useState('');
  const [hasAgreed, setHasAgreed] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (userName && hasAgreed && messages.length === 0) {
      const welcomeMessage: Message = {
        id: '1',
        text: `Welcome ${userName} to AlgoRythm AI Europa! 🚀\n\nI'm an advanced AI assistant created by AlgoRythm Tech in Hyderabad, India - the first ever teen-built AI startup, founded by CEO Sri Aasrith Souri Kompella.\n\nHow can I assist you today?`,
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
    }
  }, [userName, hasAgreed]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    try {
      const { response } = await chatAPI.sendMessage(inputText, userName);
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setUploadedFile(file);
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <h2>Loading AlgoRythm AI...</h2>
      </div>
    );
  }

  if (!userName || !hasAgreed) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <h1>AlgoRythm AI Europa</h1>
          <h3>Advanced AI Model</h3>
          <p>Created by AlgoRythm Tech, Hyderabad</p>
          <p className="subtitle">The first ever teen-built AI startup</p>
          <p className="ceo">Founded by CEO Sri Aasrith Souri Kompella</p>
          
          <input
            type="text"
            placeholder="Enter your name"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            className="name-input"
          />
          
          <label className="terms-checkbox">
            <input
              type="checkbox"
              checked={hasAgreed}
              onChange={(e) => setHasAgreed(e.target.checked)}
            />
            I agree to the Terms and Conditions
          </label>

          <button 
            className="auth-button" 
            onClick={() => setHasAgreed(true)}
            disabled={!userName || !hasAgreed}
          >
            Continue to Chat
          </button>

          <div className="features">
            <div className="feature">🚀 Advanced AI</div>
            <div className="feature">🤖 Real-time Chat</div>
            <div className="feature">🎯 Precise Responses</div>
            <div className="feature">🧠 Smart Learning</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>AlgoRythm AI Europa</h1>
        <span>Created by AlgoRythm Tech | CEO: Sri Aasrith Souri Kompella</span>
        <div className="header-actions">
          <span className="user-info">👤 {userName}</span>
          <button onClick={() => {
            setUserName('');
            setHasAgreed(false);
            setMessages([]);
          }}>
            Logout
          </button>
        </div>
      </header>

      <div className="chat-container">
        <div className="messages-area">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              <div className="message-content">
                <strong>{message.sender === 'user' ? userName : 'AlgoRythm AI'}:</strong>
                <p>{message.text}</p>
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="message ai typing">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <div className="input-controls">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask anything... (Created by AlgoRythm Tech)"
              disabled={isTyping}
              className="message-input"
            />
            <button onClick={handleSendMessage} disabled={!inputText.trim() || isTyping}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return <AlgoRythmAIApp />;
}

export default App;
