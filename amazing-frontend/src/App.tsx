
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithPopup, GoogleAuthProvider } from 'firebase/auth';
import './App.css';
import type { ChangeEvent, FormEvent } from 'react';

type Message = {
	role: 'user' | 'assistant';
	content: string;
	model: string;
};

const API_URL = 'http://localhost:8000';

const firebaseConfig = {
  apiKey: "AIzaSyCjRkPiHNp8KoNNpZELsfT1ujZ7cPz-XwE",
  authDomain: "rythm-europa.firebaseapp.com",
  projectId: "rythm-europa",
  storageBucket: "rythm-europa.appspot.com",
  messagingSenderId: "1087218166120",
  appId: "1:1087218166120:web:694f657cbfd24147580edd",
  measurementId: "G-ERDLE4THNH"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

function App() {
	const [user, setUser] = useState<any>(null);
    
	useEffect(() => {
		const unsubscribe = auth.onAuthStateChanged((user) => {
			setUser(user);
		});
		return () => unsubscribe();
	}, []);

	const signInWithGoogle = async () => {
		try {
			await signInWithPopup(auth, googleProvider);
		} catch (error) {
			console.error('Error signing in with Google:', error);
		}
	};

	const MODELS = [
		{ label: 'Rythm AI Europa', value: 'rythm' },
		{ label: 'GPT-5', value: 'gpt5' },
		{ label: 'GPT-4', value: 'gpt4' },
		{ label: 'GPT-3', value: 'gpt3' },
		{ label: 'Claude 3.5', value: 'claude35' },
		{ label: 'Claude 4 Sonnet', value: 'claude4sonnet' },
		{ label: 'Meta Llama', value: 'llama' },
		{ label: 'DeepSeek', value: 'deepseek' },
	];
	const [messages, setMessages] = useState<Message[]>([]);
	const [input, setInput] = useState<string>("");
	const [loading, setLoading] = useState<boolean>(false);
	const [error, setError] = useState<string | null>(null);
	const [selectedModel, setSelectedModel] = useState<string>('rythm');
	const [otherModelQueries, setOtherModelQueries] = useState<number>(30);
	const [backendHealthy, setBackendHealthy] = useState<boolean>(true);
	const chatBoxRef = useRef<HTMLDivElement>(null);
	const [userName, setUserName] = useState('');
	const [agreed, setAgreed] = useState(false);
	const [modelChecks, setModelChecks] = useState<{[key:string]: boolean}>({});

	useEffect(() => {
		const checkBackend = async () => {
			try {
				await axios.get(`${API_URL}/health`, { timeout: 4000 });
				setBackendHealthy(true);
			} catch {
				setBackendHealthy(false);
			}
		};
		checkBackend();
	}, []);

	// No auth effect needed

	useEffect(() => {
		const count = localStorage.getItem('otherModelQueries');
		if (count !== null) setOtherModelQueries(Number(count));
	}, []);

	useEffect(() => {
		localStorage.setItem('otherModelQueries', String(otherModelQueries));
	}, [otherModelQueries]);

	useEffect(() => {
		if (otherModelQueries <= 0 && selectedModel !== 'rythm') {
			setSelectedModel('rythm');
		}
	}, [otherModelQueries, selectedModel]);

	useEffect(() => {
		if (chatBoxRef.current) {
			chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
		}
	}, [messages]);

	const sendMessage = async (e: FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (!input.trim()) return;
		if (!user) {
			setError('Please sign in to continue.');
			return;
		}
		if (!userName || !agreed) {
			setError('Please enter your name and agree to the Terms & Conditions.');
			return;
		}
		if (selectedModel !== 'rythm' && otherModelQueries <= 0) {
			setError('You have reached your monthly limit for other models. Please use Rythm AI Europa.');
			setSelectedModel('rythm');
			return;
		}
		setLoading(true);
		setError(null);
		setMessages(msgs => [...msgs, { role: 'user', content: input, model: selectedModel }]);
		setInput("");
		try {
			const res = await axios.post(
				`${API_URL}/api/chat`,
				{
					message: input,
					model: selectedModel
				},
				{ timeout: 15000 }
			);
			setMessages(msgs => [...msgs, { role: 'assistant', content: res.data.response, model: selectedModel }]);
			if (selectedModel !== 'rythm') {
				setOtherModelQueries(q => q - 1);
			}
		} catch (err: any) {
			if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
				setError('Chat API timed out. Please try again or check backend performance.');
			} else if (err.response) {
				setError(`Chat API error: ${err.response.status} ${err.response.statusText}`);
			} else if (err.request) {
				setError('Chat API not reachable. Please ensure backend is running.');
			} else {
				setError((err && err.message) ? err.message : 'Error calling backend');
			}
		} finally {
			setLoading(false);
		}
	};

			return (
				<div className="gemini-bg">
					<div className="gemini-card">
						<header className="gemini-header">
							<div className="gemini-logo-row">
								<span className="gemini-logo">🔮</span>
								<span className="gemini-title">Rythm AI Europa</span>
							</div>
							{!user && (
								<div className="gemini-auth-box">
									<button onClick={signInWithGoogle} className="gemini-btn google">
										Sign in with Google to Continue
									</button>
								</div>
							)}
							{(!userName || !agreed) ? (
								<div className="gemini-auth-box">
									<input className="gemini-input" type="text" value={userName} onChange={e => setUserName(e.target.value)} placeholder="Enter your name" required />
									<div className="gemini-tc-box">
										<label>
											<input type="checkbox" checked={agreed} onChange={e => setAgreed(e.target.checked)} />
											I agree to the <a href="#" target="_blank" rel="noopener noreferrer">Terms & Conditions</a> and acknowledge that Rythm AI may make mistakes. Always cross-check important information.
										</label>
									</div>
									<div className="gemini-tc-box">
										<span>Select models you want to use:</span>
										{MODELS.map(m => (
											<label key={m.value} style={{marginRight:8}}>
												<input type="checkbox" checked={!!modelChecks[m.value]} onChange={e => setModelChecks(prev => ({...prev, [m.value]: e.target.checked}))} /> {m.label}
											</label>
										))}
									</div>
								</div>
							) : null}
							{(userName && agreed) && (
								<div className="gemini-model-row">
									<label htmlFor="model-select">Model:</label>
									<select
										className="gemini-select"
										id="model-select"
										value={selectedModel}
										onChange={(e: ChangeEvent<HTMLSelectElement>) => setSelectedModel(e.target.value)}
										disabled={otherModelQueries <= 0}
									>
										{MODELS.filter(m => modelChecks[m.value]).map((m) => (
											<option key={m.value} value={m.value} disabled={otherModelQueries <= 0 && m.value !== 'rythm'}>
												{m.label}
											</option>
										))}
									</select>
									<span className="gemini-model-info">
										{selectedModel === 'rythm' || otherModelQueries > 0
											? (selectedModel === 'rythm' ? 'Unlimited queries' : `${otherModelQueries} queries left this month`)
											: 'Other models disabled, use Rythm AI Europa'}
									</span>
								</div>
							)}
							{!backendHealthy && (
								<div className="gemini-error">
									Backend is not reachable. Please ensure the backend is running.
								</div>
							)}
						</header>
				{(userName && agreed) && (
					<main className="gemini-main">
						<div className="gemini-chat-box" ref={chatBoxRef}>
							{messages.map((msg, idx) => {
								let content = msg.content;
								if (msg.role === 'assistant' && content.startsWith('Wolfram Alpha says:')) {
									content = content.replace(/^Wolfram Alpha says:\s*/, '');
								}
								return (
									<div key={idx} className={msg.role === 'user' ? 'gemini-user-msg' : 'gemini-ai-msg'}>
										<div className="gemini-msg-meta">
											<b>{msg.role === 'user' ? userName : 'AI'} [{MODELS.find(m => m.value === msg.model)?.label || msg.model}]</b>
										</div>
										<div className="gemini-msg-content">{content}</div>
									</div>
								);
							})}
						</div>
						<form onSubmit={sendMessage} className="gemini-input-form">
							<input
								className="gemini-input"
								type="text"
								value={input}
								onChange={(e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
								placeholder="Type your message..."
								disabled={loading}
								autoFocus
							/>
							<button className="gemini-btn send" type="submit" disabled={loading || (!input.trim())}>
								{loading ? 'Send...' : 'Send'}
							</button>
							<button className="gemini-btn clear" type="button" onClick={() => setMessages([])} disabled={loading}>
								Clear
							</button>
						</form>
						{loading && <div className="gemini-wait">Waiting for AI response...</div>}
						{error && <div className="gemini-error">{error}</div>}
					</main>
				)}
					</div>
				</div>
			);
}

export default App;
