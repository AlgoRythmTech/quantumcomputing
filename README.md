# 🚀 RythmAI Europa - Advanced AI System

<div align="center">
  <img src="https://img.shields.io/badge/Version-1.2.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/AI%20Model-8B%20Parameters-green" alt="Model">
  <img src="https://img.shields.io/badge/License-MIT-purple" alt="License">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success" alt="Status">
</div>

<div align="center">
  <h3>Created by AlgoRythm Tech</h3>
  <p><strong>The First Ever Teen-Built AI Startup</strong></p>
  <p>CEO & Founder: Sri Aasrith Souri Kompella | Hyderabad, India</p>
</div>

---

## 🌟 Features

- **🤖 Advanced AI Chat** - Powered by state-of-the-art language models
- **🔍 Deep Web Search** - Real-time web search integration
- **📄 PDF Generation** - Generate professional PDFs from conversations
- **🖼️ Image Analysis** - Advanced computer vision capabilities
- **📁 File Processing** - Upload and process various file formats
- **🔐 Secure Authentication** - Auth0 integration for secure access
- **⚡ Real-time Response** - Fast and efficient AI responses
- **🎨 Modern UI** - Beautiful, responsive design with Material-UI

## 🛠️ Tech Stack

### Frontend
- **React 19** with TypeScript
- **Material-UI** for modern components
- **Redux Toolkit** for state management
- **Axios** for API communication
- **Auth0** for authentication
- **Vite** for blazing fast builds

### Backend
- **FastAPI** - High-performance Python framework
- **PyTorch** - Deep learning framework
- **Transformers** - State-of-the-art NLP models
- **FAISS** - Vector similarity search
- **Pillow & OpenCV** - Image processing
- **ReportLab** - PDF generation

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/RythmAi_Europa.git
cd RythmAi_Europa
```

2. **Setup Frontend**
```bash
cd frontend
npm install
cp .env.example .env
# Edit .env with your configuration
```

3. **Setup Backend**
```bash
cd backend
pip install -r requirements.txt
# Create .env file with your configuration
```

### Running Locally

**Using the deployment script (Windows):**
```powershell
./deploy.ps1
# Select option 5 for local development
```

**Or manually:**

Terminal 1 - Backend:
```bash
cd backend
python algorythm_ai_backend.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Access the application at `http://localhost:5173`

## 🌐 Deployment (100% FREE)

### Frontend Deployment

**Option 1: Vercel (Recommended)**
```bash
cd frontend
npm run deploy:vercel
```

**Option 2: Netlify**
```bash
cd frontend
npm run deploy:netlify
```

### Backend Deployment

**Deploy to Render.com (FREE)**
1. Push code to GitHub
2. Create account at [render.com](https://render.com)
3. Create new Web Service
4. Connect GitHub repository
5. Configure:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn algorythm_ai_backend:app --host 0.0.0.0 --port $PORT`

### Deployment Guide
See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed instructions.

## 📁 Project Structure

```
RythmAi_Europa/
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   ├── store/         # Redux store
│   │   └── App.tsx        # Main application
│   ├── vercel.json        # Vercel configuration
│   └── netlify.toml       # Netlify configuration
├── backend/               # FastAPI backend
│   ├── algorythm_ai_backend.py  # Main backend application
│   ├── requirements.txt  # Python dependencies
│   ├── Dockerfile        # Docker configuration
│   └── render.yaml       # Render.com configuration
├── deploy.ps1            # Deployment script
└── DEPLOYMENT_GUIDE.md   # Deployment documentation
```

## 🔐 Environment Variables

### Frontend (.env)
```env
VITE_API_URL=http://localhost:8000
VITE_AUTH0_DOMAIN=your-domain.auth0.com
VITE_AUTH0_CLIENT_ID=your-client-id
```

### Backend (.env)
```env
AUTH0_DOMAIN=your-domain.auth0.com
AUTH0_API_AUDIENCE=your-api-audience
```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check & info |
| `/api/chat` | POST | Main chat endpoint |
| `/api/analyze-image` | POST | Image analysis |
| `/api/search` | POST | Web search |
| `/api/upload` | POST | File upload |
| `/api/download/{filename}` | GET | Download PDFs |

## 📊 Performance

- **Response Time**: < 2 seconds average
- **Concurrent Users**: 100+ supported
- **Uptime**: 99.9% on production
- **Model Parameters**: 8 billion
- **Languages Supported**: 50+

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AlgoRythm Tech Team
- Open source community
- All contributors and users

## 📞 Contact & Support

- **Company**: AlgoRythm Tech
- **Location**: Hyderabad, India
- **CEO**: Sri Aasrith Souri Kompella
- **Email**: helpdeskalgt@gmail.com



---

<div align="center">
  <h3>Built with ❤️ by AlgoRythm Tech</h3>
  <p>Empowering the future with AI</p>
</div>
