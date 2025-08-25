"""
Deployment Configuration for rythm.ai
AlgoRythm Tech Production Deployment
"""

import os
from pathlib import Path

# Domain Configuration
DOMAIN = "rythm.ai"
API_DOMAIN = "api.rythm.ai"

# Server Configuration
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_PORT = 3000

# Production Settings
PRODUCTION_CONFIG = {
    "backend": {
        "host": BACKEND_HOST,
        "port": BACKEND_PORT,
        "workers": 4,  # Number of worker processes
        "log_level": "info",
        "access_log": True,
        "use_colors": True,
        "reload": False,  # Never use reload in production
        "ssl_keyfile": None,  # Add SSL certificate path if available
        "ssl_certfile": None,  # Add SSL certificate path if available
    },
    "frontend": {
        "build_command": "npm run build",
        "serve_command": "npm run preview",
        "api_url": f"https://{API_DOMAIN}",
        "auth0_domain": "your-auth0-domain.auth0.com",
        "auth0_client_id": "your-client-id"
    },
    "cors": {
        "origins": [
            f"https://{DOMAIN}",
            f"https://www.{DOMAIN}",
            f"https://{API_DOMAIN}",
            "http://localhost:3000",  # Development
            "http://localhost:5173"   # Vite development
        ]
    }
}

# Environment Variables for Production
PRODUCTION_ENV = {
    "VITE_API_URL": f"https://{API_DOMAIN}",
    "VITE_AUTH0_DOMAIN": "your-auth0-domain.auth0.com",
    "VITE_AUTH0_CLIENT_ID": "your-client-id",
    "NODE_ENV": "production"
}

def create_env_files():
    """Create .env files for production deployment"""
    
    # Backend .env
    backend_env = Path("backend/.env")
    with open(backend_env, "w") as f:
        f.write(f"DOMAIN={DOMAIN}\n")
        f.write(f"API_DOMAIN={API_DOMAIN}\n")
        f.write("AUTH0_DOMAIN=your-auth0-domain.auth0.com\n")
        f.write("AUTH0_API_AUDIENCE=your-api-audience\n")
        f.write("SECRET_KEY=your-secret-key-here\n")
        f.write("ENVIRONMENT=production\n")
    
    # Frontend .env.production
    frontend_env = Path("frontend/.env.production")
    with open(frontend_env, "w") as f:
        f.write(f"VITE_API_URL=https://{API_DOMAIN}\n")
        f.write("VITE_AUTH0_DOMAIN=your-auth0-domain.auth0.com\n")
        f.write("VITE_AUTH0_CLIENT_ID=your-client-id\n")
    
    print("✅ Environment files created for production")

if __name__ == "__main__":
    create_env_files()
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              AlgoRythm AI Deployment Configuration          ║
║                                                              ║
║  Domain: {DOMAIN:<50} ║
║  API Domain: {API_DOMAIN:<46} ║
║                                                              ║
║  Backend Port: {BACKEND_PORT:<44} ║
║  Frontend Port: {FRONTEND_PORT:<43} ║
║                                                              ║
║  Ready for deployment to production!                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
