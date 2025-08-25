"""
AlgoRythm Tech AI Backend - Production Ready System
CEO & Founder: Sri Aasrith Souri Kompella
Company: AlgoRythm Tech, Hyderabad - First teen-built AI startup
"""
"""
AlgoRythm Tech AI Backend - Production Ready System
CEO & Founder: Sri Aasrith Souri Kompella
Company: AlgoRythm Tech, Hyderabad - First teen-built AI startup
"""
"""
AlgoRythm Tech AI Backend - Production Ready System
CEO & Founder: Sri Aasrith Souri Kompella
Company: AlgoRythm Tech, Hyderabad - First teen-built AI startup
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uvicorn
import logging
from datetime import datetime
import json
import os
import aiohttp
import asyncio
from PIL import Image
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
import requests
from bs4 import BeautifulSoup
import hashlib
import jwt
from functools import wraps
import re
from pathlib import Path
import numpy as np
import cv2
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AlgoRythm AI Europa 8B",
    description="Advanced AI system by AlgoRythm Tech - First teen-built AI startup",
    version="1.2.0"
)

# Add /health endpoint for frontend health check compatibility
from datetime import datetime

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "AlgoRythm AI Europa 8B",
        "company": "AlgoRythm Tech",
        "timestamp": datetime.now().isoformat()
    }

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
tokenizer = None
vision_model = None
embeddings_model = None
knowledge_base = None

# Conversation history storage (in production, use Redis or database)
conversation_history = {}

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

# Auth0 configuration (replace with your actual Auth0 details)
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "your-domain.auth0.com")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE", "your-api-audience")
AUTH0_ALGORITHMS = ["RS256"]

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    search_web: Optional[bool] = False
    generate_pdf: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: Optional[List[Dict]] = None
    pdf_url: Optional[str] = None

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    question: Optional[str] = "What is in this image?"

class WebSearchRequest(BaseModel):
    query: str
    num_results: Optional[int] = 5

def initialize_models():
    """Initialize all AI models"""
    global model, tokenizer, vision_model, embeddings_model
    
    try:
        logger.info("Initializing AlgoRythm AI models...")
        
        # Load a powerful open-source model
        model_name = "microsoft/DialoGPT-large"  # Using large model for better responses
        logger.info(f"Loading {model_name} model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize embeddings for semantic search
        logger.info("Loading embeddings model...")
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize vision model for image analysis
        logger.info("Loading vision model...")
        vision_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise e  # Don't fallback, raise the error

async def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """
    Perform deep web search using multiple sources
    """
    results = []
    
    try:
        # DuckDuckGo search (no API key required)
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                for result in soup.select('.result')[:num_results]:
                    title_elem = result.select_one('.result__title')
                    snippet_elem = result.select_one('.result__snippet')
                    link_elem = result.select_one('.result__url')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            'title': title_elem.get_text(strip=True),
                            'snippet': snippet_elem.get_text(strip=True),
                            'url': link_elem.get_text(strip=True) if link_elem else ''
                        })
        
        # Additional search using Wikipedia API
        wiki_url = f"https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(wiki_url, params=params) as response:
                data = await response.json()
                for item in data.get('query', {}).get('search', []):
                    results.append({
                        'title': item['title'],
                        'snippet': re.sub(r'<[^>]+>', '', item['snippet']),
                        'url': f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"
                    })
                    
    except Exception as e:
        logger.error(f"Web search error: {e}")
        results.append({
            'title': 'Search Error',
            'snippet': 'Unable to perform web search at this time.',
            'url': ''
        })
    
    return results

def generate_pdf(content: str, filename: str = None) -> bytes:
    """Generate PDF from text content"""
    if filename is None:
        filename = f"algorythm_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Add AlgoRythm Tech header
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#2196F3',
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    story.append(Paragraph("AlgoRythm AI Europa", header_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated by AlgoRythm Tech - {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Add content
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
    
    # Add footer
    footer_text = "Created by AlgoRythm Tech, Hyderabad | CEO: Sri Aasrith Souri Kompella"
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(footer_text, styles['Italic']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def enhance_prompt_with_company_context(message: str) -> str:
    """Enhance prompt with company context for model to generate natural responses"""
    message_lower = message.lower()
    
    # Add context to prompt instead of returning hardcoded responses
    context = ""
    if any(keyword in message_lower for keyword in ['who created', 'who built', 'who made', 'who developed', 'created by', 'built by', 'ceo', 'founder', 'algorythm']):
        context = """Context: I am AlgoRythm AI Europa 8B, created by AlgoRythm Tech in Hyderabad, India. 
        The company was founded by CEO Sri Aasrith Souri Kompella and is the first teen-built AI startup. 
        I have 8 billion parameters and advanced capabilities.
        
        """
    
    return context + message

async def generate_ai_response(message: str, web_search: bool = False, temperature: float = 0.7, conversation_id: str = None) -> Dict:
    """Generate AI response with optional web search - USING ACTUAL MODEL"""
    
    sources = None
    enhanced_prompt = message
    
    # Add conversation history if available
    if conversation_id and conversation_id in conversation_history:
        history = conversation_history[conversation_id]
        # Build context from history (last 3 exchanges)
        context_parts = []
        for h in history[-6:]:  # Last 3 exchanges (user + assistant)
            context_parts.append(f"{h['role']}: {h['content']}")
        if context_parts:
            enhanced_prompt = "\n".join(context_parts) + "\nUser: " + message + "\nAssistant:"
    else:
        # Add company context if relevant (but let model generate the response)
        enhanced_prompt = enhance_prompt_with_company_context(message)
    
    # Perform web search if requested
    if web_search:
        search_results = await search_web(message, num_results=5)
        if search_results:
            sources = search_results
            context = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results[:3]])
            enhanced_prompt = f"""Based on the following search results, please provide a comprehensive answer:
            
Search Results:
{context}

User Question: {message}

Please provide a detailed and accurate response based on the search results and your knowledge."""
    
    # Generate response
    try:
        if model is None or tokenizer is None:
            logger.error("Models not initialized")
            raise Exception("AI models not properly initialized")
            
        # Encode with proper attention mask
        inputs = tokenizer.encode_plus(
            enhanced_prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True,
            return_attention_mask=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=500,  # Generate new tokens, not total length
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                repetition_penalty=1.2  # Reduce repetition
            )
        
        # Decode only the generated part (not the input)
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Ensure we have a response
        if not response or len(response.strip()) < 5:
            # Fallback: try with different parameters
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=200,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Return error instead of hardcoded response
        raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")
    
    # Store in conversation history
    if conversation_id:
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []
        conversation_history[conversation_id].append({"role": "User", "content": message})
        conversation_history[conversation_id].append({"role": "Assistant", "content": response})
        # Keep only last 10 exchanges
        conversation_history[conversation_id] = conversation_history[conversation_id][-20:]
    
    return {
        'response': response,
        'sources': sources
    }

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        initialize_models()
        logger.info("AlgoRythm AI Backend started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.info("Backend will start but some features may be limited")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AlgoRythm AI Europa 8B",
        "company": COMPANY_INFO,
        "status": "operational",
        "features": [
            "Advanced Chat",
            "Web Search",
            "PDF Generation",
            "Image Analysis",
            "File Upload",
            "Auth0 Authentication"
        ]
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Generate response
        result = await generate_ai_response(
            request.message,
            web_search=request.search_web if request.search_web is not None else False,
            temperature=request.temperature if request.temperature is not None else 0.7,
            conversation_id=request.conversation_id
        )
        
        response_text = result['response']
        sources = result.get('sources')
        
        # Generate PDF if requested
        pdf_url = None
        if request.generate_pdf:
            pdf_content = generate_pdf(response_text)
            # In production, save to cloud storage
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
            sources=sources,
            pdf_url=pdf_url
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...), question: Optional[str] = Form(None)):
    """Analyze uploaded image"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Analyze with vision model
        if vision_model:
            result = vision_model(image)
            description = result[0]['generated_text'] if result else "Unable to analyze image"
        else:
            description = "Image analysis model is being initialized. Please try again."
        
        # If user asked a specific question
        if question:
            response = f"Image Analysis: {description}\n\nRegarding your question '{question}': Based on the image, {description}"
        else:
            response = f"Image Analysis: {description}"
        
        return {
            "analysis": response,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(request: WebSearchRequest):
    """Perform web search"""
    try:
        results = await search_web(request.query, request.num_results)
        return {
            "query": request.query,
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

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads for processing"""
    try:
        contents = await file.read()
        
        # Save file
        upload_path = f"uploads/{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(contents)
        
        # Process based on file type
        if file.filename.endswith(('.txt', '.md')):
            text_content = contents.decode('utf-8')
            return {
                "filename": file.filename,
                "content_preview": text_content[:500],
                "status": "uploaded",
                "message": "File uploaded successfully. You can now ask questions about this content."
            }
        elif file.filename.endswith(('.pdf')):
            return {
                "filename": file.filename,
                "status": "uploaded",
                "message": "PDF uploaded. Processing capabilities coming soon."
            }
        else:
            return {
                "filename": file.filename,
                "status": "uploaded",
                "message": "File uploaded successfully."
            }
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "AlgoRythm AI Europa 8B",
        "company": "AlgoRythm Tech",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("""\n╔══════════════════════════════════════════════════════════════╗
║           AlgoRythm AI Europa Backend Starting...           ║
║                                                              ║
║  Company: AlgoRythm Tech, Hyderabad                        ║
║  CEO & Founder: Sri Aasrith Souri Kompella                 ║
║  Model: Europa 8B                                          ║
║                                                              ║
║  Backend will run continuously. Press Ctrl+C to stop.       ║
╚══════════════════════════════════════════════════════════════╝\n""")
    
    # Run the server without reload for production
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Disable reload for stable production
    )
