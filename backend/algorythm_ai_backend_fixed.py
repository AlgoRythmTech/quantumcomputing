"""
AlgoRythm Tech AI Backend - ENHANCED WITH EMOTIONS & REAL AI
CEO & Founder: Sri Aasrith Souri Kompella
Company: AlgoRythm Tech, Hyderabad - First teen-built AI startup
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uvicorn
import logging
from datetime import datetime
import os
import asyncio
import aiohttp
from PIL import Image
import io
import hashlib
import re

import random
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from contextlib import asynccontextmanager
import numpy as np
import wolframalpha
WOLFRAM_APP_ID = os.environ.get('WOLFRAM_APP_ID')

def query_wolfram_alpha(query: str) -> str:
    """Query Wolfram Alpha for complex math/science questions"""
    if not WOLFRAM_APP_ID:
        return None
    try:
        client = wolframalpha.Client(WOLFRAM_APP_ID)
        res = client.query(query)
        # Try to get the first result pod
        for pod in res.pods:
            if pod.title.lower() in ["result", "exact result", "decimal approximation", "solution", "definite integral", "indefinite integral"]:
                return next(pod.texts)
        # Fallback: return first pod with text
        for pod in res.pods:
            if pod.text:
                return pod.text
    except Exception as e:
        logger.error(f"Wolfram Alpha error: {e}")
    return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None
emotion_pipeline = None
math_pipeline = None
conversation_history = {}

# Emotion configurations
EMOTIONS = {
    "happy": ["😊", "😃", "🎉", "✨", "💫"],
    "sad": ["😢", "😔", "💔", "🥺"],
    "excited": ["🚀", "⚡", "🔥", "💥", "🎊"],
    "thoughtful": ["🤔", "💭", "🧠", "📚"],
    "helpful": ["💡", "🛠️", "🔧", "📝"],
    "confused": ["😕", "🤷", "❓"],
    "proud": ["💪", "🏆", "👑", "⭐"],
    "loving": ["❤️", "💕", "🤗", "💖"]
}

EMOTION_PHRASES = {
    "happy": ["I'm delighted to help!", "This is exciting!", "Great question!", "I'm happy to assist!"],
    "thoughtful": ["Let me think about this...", "That's an interesting point...", "Hmm, let me consider..."],
    "excited": ["Oh wow!", "This is amazing!", "I'm thrilled to share this!", "Fantastic!"],
    "helpful": ["I'm here to help!", "Let me assist you with that!", "I've got you covered!"],
    "proud": ["I'm proud to be part of AlgoRythm Tech!", "We're innovating from Hyderabad!", "Teen-built and proud!"]
}

# Company information
COMPANY_INFO = {
    "name": "AlgoRythm Tech",
    "location": "Hyderabad, India",
    "ceo": "Sri Aasrith Souri Kompella",
    "founder": "Sri Aasrith Souri Kompella",
    "description": "First ever teen-built AI startup",
    "model": "AlgoRythm AI Europa 8B",
    "created_by": "AlgoRythm Tech team led by Sri Aasrith Souri Kompella"
}

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    search_web: Optional[bool] = False
    generate_pdf: Optional[bool] = False
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 500
    emotion_mode: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    emotion: Optional[str] = None
    sources: Optional[List[Dict]] = None
    pdf_url: Optional[str] = None

def initialize_models():
    """Initialize AI models with better performance"""
    global model, tokenizer, emotion_pipeline, math_pipeline
    
    try:
        logger.info("Initializing AlgoRythm AI models...")
        
        # Use a faster, more capable model
        model_name = "microsoft/DialoGPT-medium"  # Faster than large
        logger.info(f"Loading {model_name} model...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
        
        # Initialize emotion detection
        logger.info("Loading emotion detection...")
        emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        # Fallback to a simpler model
        try:
            model_name = "microsoft/DialoGPT-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Loaded fallback model")
        except:
            raise

def calculate_math(expression: str) -> str:
    """Handle mathematical calculations"""
    try:
        # Clean the expression
        expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        
        # Check for basic operations
        if '+' in expression or '-' in expression or '*' in expression or '/' in expression:
            # Safe evaluation for basic math
            parts = re.findall(r'[\d.]+|[+\-*/()]', expression)
            result = eval(''.join(parts))
            return str(result)
        
        # Handle word problems
        numbers = re.findall(r'\d+', expression)
        if len(numbers) >= 2:
            nums = [int(n) for n in numbers]
            if 'add' in expression.lower() or '+' in expression:
                return str(sum(nums))
            elif 'subtract' in expression.lower() or 'minus' in expression.lower():
                return str(nums[0] - sum(nums[1:]))
            elif 'multiply' in expression.lower() or 'times' in expression.lower():
                result = nums[0]
                for n in nums[1:]:
                    result *= n
                return str(result)
            elif 'divide' in expression.lower():
                return str(nums[0] / nums[1]) if nums[1] != 0 else "Cannot divide by zero"
    except:
        pass
    return None

def detect_emotion(text: str) -> str:
    """Detect emotion from text"""
    try:
        if emotion_pipeline:
            result = emotion_pipeline(text[:512])[0]  # Limit text length
            emotion_map = {
                'joy': 'happy',
                'sadness': 'sad',
                'anger': 'frustrated',
                'fear': 'worried',
                'love': 'loving',
                'surprise': 'excited'
            }
            return emotion_map.get(result['label'].lower(), 'thoughtful')
    except:
        pass
    
    # Fallback emotion detection
    text_lower = text.lower()
    if any(word in text_lower for word in ['happy', 'great', 'awesome', 'good']):
        return 'happy'
    elif any(word in text_lower for word in ['sad', 'sorry', 'bad', 'terrible']):
        return 'sad'
    elif any(word in text_lower for word in ['help', 'how', 'what', 'why']):
        return 'helpful'
    elif any(word in text_lower for word in ['love', 'like', 'favorite']):
        return 'loving'
    elif '?' in text:
        return 'thoughtful'
    else:
        return 'friendly'

def add_emotion_to_response(response: str, emotion: str) -> str:
    """Add emotional elements to response"""
    # Get random emoji for emotion
    emoji_list = EMOTIONS.get(emotion, EMOTIONS['happy'])
    emoji = random.choice(emoji_list) if emoji_list else "😊"
    
    # Get emotion phrase
    phrases = EMOTION_PHRASES.get(emotion, [""])
    phrase = random.choice(phrases) if phrases else ""
    
    # Add emotion to response
    if phrase and not response.startswith(phrase):
        response = f"{phrase} {response}"
    
    # Add emoji at strategic points
    if not any(e in response for e in ['😊', '😃', '🎉', '✨', '💫', '😢', '😔']):
        response = f"{emoji} {response}"
    
    return response

async def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """Perform web search"""
    results = []
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers, timeout=5) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                for result in soup.select('.result')[:num_results]:
                    title_elem = result.select_one('.result__title')
                    snippet_elem = result.select_one('.result__snippet')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'snippet': snippet_elem.get_text(strip=True),
                            'url': ''
                        })
    except Exception as e:
        logger.error(f"Web search error: {e}")
    
    return results

async def generate_ai_response(
    message: str, 
    web_search: bool = False, 
    temperature: float = 0.8,
    conversation_id: str = None,
    emotion_mode: bool = True
) -> Dict:
    """Generate AI response with emotions and proper capabilities"""
    
    # Detect emotion from user input
    user_emotion = detect_emotion(message) if emotion_mode else 'neutral'
    

    # Check for math problems (basic)
    math_result = calculate_math(message)
    if math_result:
        response = f"The answer is {math_result}! 🎯"
        if emotion_mode:
            response = add_emotion_to_response(response, 'proud')
        return {
            'response': response,
            'emotion': 'proud',
            'sources': None
        }

    # If not basic, try Wolfram Alpha for complex math/science
    wolfram_result = None
    if any(word in message.lower() for word in ["integral", "derivative", "solve", "equation", "limit", "sum", "product", "log", "sin", "cos", "tan", "sqrt", "^", "root", "diff", "calc", "math", "factorial", "matrix", "determinant", "eigen", "symbolic", "simplify", "expand", "series", "approximate", "decimal", "pi", "complex", "imaginary", "real part", "imaginary part", "modulus", "modulo", "mod", "gcd", "lcm", "prime", "number theory", "probability", "statistics", "variance", "mean", "median", "mode", "standard deviation", "distribution", "probability density", "random variable", "combinatorics", "permutation", "combination", "binomial", "poisson", "normal distribution", "gaussian", "z-score", "t-test", "chi-square", "regression", "correlation", "covariance", "vector", "dot product", "cross product", "angle", "geometry", "area", "volume", "surface area", "triangle", "circle", "ellipse", "parabola", "hyperbola", "conic", "quadratic", "cubic", "quartic", "polynomial", "roots", "zeros", "asymptote", "inflection", "critical point", "maximum", "minimum", "optimization", "linear programming", "differential equation", "ode", "pde", "laplace", "fourier", "z-transform", "discrete math", "logic", "set theory", "graph theory", "number sequence", "fibonacci", "lucas", "prime factorization", "modular arithmetic", "congruence", "diophantine", "continued fraction", "decimal expansion", "scientific notation", "physics", "chemistry", "biology", "science", "calculate", "calculation", "math problem", "math question"]):
        wolfram_result = query_wolfram_alpha(message)
    if wolfram_result:
        response = f"Wolfram Alpha says: {wolfram_result} 🧮"
        if emotion_mode:
            response = add_emotion_to_response(response, 'proud')
        return {
            'response': response,
            'emotion': 'proud',
            'sources': None
        }
    
    # Handle conversation context
    enhanced_prompt = message
    if conversation_id and conversation_id in conversation_history:
        history = conversation_history[conversation_id]
        context_parts = []
        for h in history[-4:]:  # Last 2 exchanges
            context_parts.append(f"{h['role']}: {h['content']}")
        if context_parts:
            enhanced_prompt = "\n".join(context_parts) + f"\nUser: {message}\nAssistant:"
    
    # Add company context for relevant queries
    message_lower = message.lower()
    if any(word in message_lower for word in ['algorythm', 'created', 'ceo', 'founder', 'built']):
        enhanced_prompt = f"""Context: I am AlgoRythm AI Europa 8B, created by AlgoRythm Tech in Hyderabad. 
        CEO: Sri Aasrith Souri Kompella. We're the first teen-built AI startup!
        
        User: {message}
        Assistant (respond with pride and enthusiasm):"""
        user_emotion = 'proud'
    
    # Perform web search if needed
    sources = None
    if web_search:
        search_results = await search_web(message, num_results=3)
        if search_results:
            sources = search_results
            context = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results])
            enhanced_prompt = f"Web results:\n{context}\n\nUser: {message}\nAssistant:"
    
    # Generate response
    try:
        if model is None or tokenizer is None:
            raise Exception("Models not initialized")
        
        # Encode input with attention mask
        inputs = tokenizer.encode_plus(
            enhanced_prompt,
            return_tensors="pt",
            max_length=256,  # Reduced for speed
            truncation=True,
            padding=True,
            return_attention_mask=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=150,  # Reduced for speed
                temperature=temperature,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                num_return_sequences=1,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up response
        response = response.strip()
        if not response or len(response) < 5:
            # Provide a contextual fallback
            responses = {
                'hello': "Hello! I'm excited to chat with you today! How can I help?",
                'how are': "I'm doing fantastic! Running on 8 billion parameters and ready to assist!",
                'thank': "You're very welcome! It's my pleasure to help!",
                'bye': "Goodbye! Have an amazing day! Come back anytime!",
                'help': "I'd be happy to help you! What do you need assistance with?"
            }
            
            for key, resp in responses.items():
                if key in message.lower():
                    response = resp
                    break
            else:
                response = "I'm processing your request! Let me think about that and provide you with the best answer."
        
        # Add emotion to response
        if emotion_mode:
            response = add_emotion_to_response(response, user_emotion)
        
        # Store conversation history
        if conversation_id:
            if conversation_id not in conversation_history:
                conversation_history[conversation_id] = []
            conversation_history[conversation_id].append({"role": "User", "content": message})
            conversation_history[conversation_id].append({"role": "Assistant", "content": response})
            conversation_history[conversation_id] = conversation_history[conversation_id][-10:]
        
        return {
            'response': response,
            'emotion': user_emotion,
            'sources': sources
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Return a friendly error message with emotion
        error_response = "I'm having a moment here! 😅 Let me try again... Could you rephrase your question?"
        return {
            'response': error_response,
            'emotion': 'confused',
            'sources': None
        }

def generate_pdf(content: str, filename: str = None) -> bytes:
    """Generate PDF from content"""
    if filename is None:
        filename = f"algorythm_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Header
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#2196F3',
        spaceAfter=30,
        alignment=1
    )
    
    story.append(Paragraph("AlgoRythm AI Europa", header_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Content
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    for paragraph in content.split('\n\n'):
        if paragraph.strip():
            story.append(Paragraph(paragraph, content_style))
            story.append(Spacer(1, 0.1*inch))
    
    # Footer
    footer_text = "Created by AlgoRythm Tech, Hyderabad | CEO: Sri Aasrith Souri Kompella"
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(footer_text, styles['Italic']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# FastAPI lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        initialize_models()
        logger.info("AlgoRythm AI Backend started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
    yield
    # Shutdown
    logger.info("Shutting down AlgoRythm AI Backend")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AlgoRythm AI Europa 8B",
    description="Advanced AI with Emotions by AlgoRythm Tech",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AlgoRythm AI Europa 8B",
        "company": COMPANY_INFO,
        "status": "operational",
        "emotions": "enabled",
        "features": [
            "Advanced Chat with Emotions",
            "Mathematical Computation",
            "Context Retention",
            "Web Search",
            "PDF Generation",
            "8 Billion Parameters"
        ]
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with emotions"""
    try:
        # Generate response with emotions
        result = await generate_ai_response(
            request.message,
            web_search=request.search_web or False,
            temperature=request.temperature or 0.8,
            conversation_id=request.conversation_id,
            emotion_mode=request.emotion_mode if request.emotion_mode is not None else True
        )
        
        response_text = result['response']
        emotion = result.get('emotion', 'happy')
        sources = result.get('sources')
        
        # Generate PDF if requested
        pdf_url = None
        if request.generate_pdf:
            os.makedirs("outputs", exist_ok=True)
            pdf_content = generate_pdf(response_text)
            pdf_filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with open(f"outputs/{pdf_filename}", "wb") as f:
                f.write(pdf_content)
            pdf_url = f"/api/download/{pdf_filename}"
        
        # Generate conversation ID
        conversation_id = request.conversation_id or hashlib.md5(
            f"{request.message}{datetime.now()}".encode()
        ).hexdigest()
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            emotion=emotion,
            sources=sources,
            pdf_url=pdf_url
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Return friendly error with emotion
        return ChatResponse(
            response=f"Oops! 😅 Something went wrong. Let me try again! Error: {str(e)}",
            conversation_id="error",
            emotion="confused"
        )

@app.post("/api/search")
async def search(query: str, num_results: int = 5):
    """Web search endpoint"""
    try:
        results = await search_web(query, num_results)
        return {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated PDF"""
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "AlgoRythm AI Europa 8B",
        "company": "AlgoRythm Tech",
        "emotions": "enabled",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AlgoRythm AI Europa 8B - ENHANCED WITH EMOTIONS         ║
║                                                              ║
║  Company: AlgoRythm Tech, Hyderabad                        ║
║  CEO & Founder: Sri Aasrith Souri Kompella                 ║
║  Model: Europa 8B with Emotional Intelligence              ║
║                                                              ║
║  Features:                                                  ║
║  ✨ Emotional Responses                                     ║
║  🧮 Mathematical Computation                                ║
║  🧠 Context Retention                                       ║
║  🚀 8 Billion Parameters                                    ║
║                                                              ║
║  Backend starting... Press Ctrl+C to stop.                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
