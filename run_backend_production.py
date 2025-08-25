#!/usr/bin/env python
"""
AlgoRythm AI Europa Production Backend Runner
This script runs the backend continuously until manually stopped with Ctrl+C
Company: AlgoRythm Tech, Hyderabad
CEO & Founder: Sri Aasrith Souri Kompella
"""

import uvicorn
import sys
import os
import signal
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlgoRythm-Backend")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n╔══════════════════════════════════════════════════════════════╗")
    print("║         AlgoRythm AI Backend Shutting Down...               ║")
    print("║         Thank you for using AlgoRythm Tech!                 ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    sys.exit(0)

def main():
    """Main function to run the backend"""
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Change to backend directory
    os.chdir(backend_path)
    
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         AlgoRythm AI Europa - Production Backend            ║
║                                                              ║
║  Company: AlgoRythm Tech, Hyderabad                        ║
║  CEO & Founder: Sri Aasrith Souri Kompella                 ║
║  Model: Europa 8B - Advanced AI System                      ║
║                                                              ║
║  🚀 Starting backend server...                              ║
║  ⚡ Backend will run continuously                           ║
║  🛑 Press Ctrl+C to stop the server                        ║
║                                                              ║
║  Access Points:                                             ║
║  - API: http://localhost:8000                              ║
║  - Docs: http://localhost:8000/docs                        ║
║  - Health: http://localhost:8000/api/health                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("Initializing AlgoRythm AI Backend...")
    
    try:
        # Run uvicorn with production settings
        uvicorn.run(
            "algorythm_ai_backend:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            use_colors=True,
            reload=False,  # NEVER use reload in production
            workers=1,  # Single worker for now, increase for production
            loop="auto",
            server_header=False,
            date_header=True,
            limit_concurrency=1000,
            limit_max_requests=10000,
            timeout_keep_alive=5
        )
    except KeyboardInterrupt:
        logger.info("Backend shutdown requested")
    except Exception as e:
        logger.error(f"Backend error: {e}")
        raise

if __name__ == "__main__":
    main()
