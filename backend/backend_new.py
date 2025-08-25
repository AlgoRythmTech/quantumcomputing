from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import torch
import logging
from pathlib import Path

# Import custom model architecture and tokenizer
from .rythm_model_architecture import RythmForCausalLM, RythmConfig
from .tokenizer_system import RythmTokenizer, TokenizerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlgoRythm AI Europa 8B",
    description="Advanced AI system by AlgoRythm Tech",
    version="1.2.0"
)

# Global variables for model
model = None
tokenizer = None

def initialize_models():
    """Initialize Rythm AI 1.2 Europa - 8B parameter model"""
    global model, tokenizer
    try:
        logger.info("Loading Rythm AI 1.2 Europa model...")
        
        # Initialize model configuration
        model_config = RythmConfig(
            vocab_size=128000,
            hidden_size=5120,
            intermediate_size=14336,
            num_hidden_layers=48,
            num_attention_heads=40,
            max_position_embeddings=32768
        )
        
        # Create model instance
        model = RythmForCausalLM(model_config)
        
        # Load trained weights if they exist
        checkpoint_path = Path("checkpoints/final_model/model.pt")
        if checkpoint_path.exists():
            logger.info(f"Loading model weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(state_dict)
        else:
            logger.warning("No pre-trained weights found, using initialized weights")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Initialize tokenizer
        tokenizer_config = TokenizerConfig(
            vocab_size=128000,
            model_type="sentencepiece",
            model_file="tokenizer/rythm_tokenizer.model"
        )
        tokenizer = RythmTokenizer(tokenizer_config)
        
        logger.info("Rythm AI model and tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()

# Configure CORS
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative dev port
    "https://rythmai.vercel.app",  # Production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    content: str
    user_name: str = "User"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="AI model not initialized")
    return {"status": "healthy", "timestamp": str(datetime.now())}

@app.post("/chat")
async def chat(message: Message):
    """Chat endpoint that processes messages and returns AI responses using Rythm AI"""
    try:
        # Prepare input with appropriate special tokens
        input_text = f"[USER]{message.user_name}: {message.content}[/USER][ASSISTANT]"
        
        # Tokenize input
        encoded = tokenizer.encode(
            input_text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        )
        
        # Move to same device as model
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device) if "attention_mask" in encoded else None
        
        # Generate response with Rythm AI
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=4096,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                num_return_sequences=1,
            )
        
        # Decode response
        response = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Clean up response
        ai_response = response.split("[ASSISTANT]")[-1].strip()
        if ai_response.endswith("[/ASSISTANT]"):
            ai_response = ai_response[:-12].strip()
        
        return {
            "response": ai_response,
            "timestamp": str(datetime.now())
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "AlgoRythm AI Europa 8B",
        "company": "AlgoRythm Tech",
        "timestamp": datetime.now().isoformat()
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model: str = "rythm"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    return {"response": f"You said: {request.message}"}
