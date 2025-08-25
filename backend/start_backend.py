"""
Start AlgoRythm AI Backend Server
"""
import uvicorn
import os

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Run the server without reload in production mode
    uvicorn.run(
        "algorythm_ai_backend:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
