# 🚀 RythmAI Europa - Deployment Guide

**Created by AlgoRythm Tech - CEO: Sri Aasrith Souri Kompella**  
*The First Ever Teen-Built AI Startup from Hyderabad, India*

## 📋 Table of Contents
1. [Overview](#overview)
2. [Frontend Deployment](#frontend-deployment)
3. [Backend Deployment](#backend-deployment)
4. [Environment Variables](#environment-variables)
5. [Post-Deployment Steps](#post-deployment-steps)

## 🎯 Overview

This guide will help you deploy RythmAI Europa completely **FREE** using:
- **Frontend**: Vercel or Netlify (both offer free hosting)
- **Backend**: Render.com (free tier with 750 hours/month)

## 🎨 Frontend Deployment

### Option 1: Deploy to Vercel (Recommended)

1. **Create Vercel Account**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub

2. **Deploy via CLI**
   ```bash
   cd frontend
   npm install -g vercel
   vercel
   ```

3. **Or Deploy via GitHub**
   - Push your code to GitHub
   - Import project in Vercel dashboard
   - It will auto-detect Vite configuration

4. **Configure Environment Variables**
   In Vercel Dashboard → Settings → Environment Variables:
   ```
   VITE_API_URL=https://your-backend-url.onrender.com
   VITE_AUTH0_DOMAIN=your-domain.auth0.com
   VITE_AUTH0_CLIENT_ID=your-client-id
   ```

### Option 2: Deploy to Netlify

1. **Create Netlify Account**
   - Go to [netlify.com](https://netlify.com)
   - Sign up with GitHub

2. **Deploy via Drag & Drop**
   ```bash
   cd frontend
   npm run build
   # Drag the 'dist' folder to Netlify dashboard
   ```

3. **Or Deploy via CLI**
   ```bash
   npm install -g netlify-cli
   netlify deploy --prod --dir=dist
   ```

4. **Configure Environment Variables**
   In Netlify Dashboard → Site Settings → Environment Variables

## 🔧 Backend Deployment

### Deploy to Render.com (FREE)

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Prepare Your Repository**
   - Push backend code to GitHub
   - Ensure `requirements.txt` is in root of backend folder

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the backend directory

4. **Configure Service**
   ```yaml
   Name: rythmAI-europa-backend
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn algorythm_ai_backend:app --host 0.0.0.0 --port $PORT
   Instance Type: Free
   ```

5. **Add Environment Variables**
   ```
   AUTH0_DOMAIN=your-domain.auth0.com
   AUTH0_API_AUDIENCE=your-api-audience
   PYTHON_VERSION=3.10
   ```

### Alternative: Deploy to Railway.app

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy from GitHub**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and deploy
   railway login
   railway up
   ```

## 🔐 Environment Variables

### Frontend (.env)
```env
VITE_API_URL=https://rythmAI-europa-backend.onrender.com
VITE_AUTH0_DOMAIN=dev-algorythm.us.auth0.com
VITE_AUTH0_CLIENT_ID=your-client-id
VITE_ENABLE_WEB_SEARCH=true
VITE_ENABLE_PDF_GENERATION=true
VITE_ENABLE_IMAGE_ANALYSIS=true
```

### Backend (.env)
```env
PORT=8000
AUTH0_DOMAIN=dev-algorythm.us.auth0.com
AUTH0_API_AUDIENCE=https://algorythm-ai-api
MODEL_CACHE_DIR=/tmp/models
```

## 🎯 Auth0 Setup (Optional but Recommended)

1. **Create Auth0 Account**
   - Go to [auth0.com](https://auth0.com)
   - Sign up for free account

2. **Create Application**
   - Applications → Create Application
   - Choose "Single Page Application"
   - Note down Domain and Client ID

3. **Configure Settings**
   - Allowed Callback URLs: `https://your-frontend-url.vercel.app`
   - Allowed Logout URLs: `https://your-frontend-url.vercel.app`
   - Allowed Web Origins: `https://your-frontend-url.vercel.app`

## 📝 Post-Deployment Steps

1. **Test All Features**
   - ✅ Chat functionality
   - ✅ Web search
   - ✅ PDF generation
   - ✅ Image analysis
   - ✅ File upload

2. **Monitor Performance**
   - Check Render.com dashboard for backend metrics
   - Monitor Vercel/Netlify analytics

3. **Custom Domain (Optional)**
   - Add custom domain in Vercel/Netlify settings
   - Update CORS settings in backend

## 🆓 Cost Breakdown

| Service | Free Tier Limits | Cost |
|---------|-----------------|------|
| Vercel | 100GB bandwidth/month | $0 |
| Netlify | 100GB bandwidth/month | $0 |
| Render.com | 750 hours/month | $0 |
| Auth0 | 7,000 active users | $0 |
| **Total** | - | **$0** |

## 🚀 Quick Deploy Scripts

### One-Click Deploy Script
```bash
# Save this as deploy.sh
#!/bin/bash

echo "🚀 Deploying RythmAI Europa..."
echo "Created by AlgoRythm Tech - CEO: Sri Aasrith Souri Kompella"

# Deploy Frontend
cd frontend
npm install
npm run build
vercel --prod

# Deploy Backend
cd ../backend
git add .
git commit -m "Deploy to Render"
git push origin main

echo "✅ Deployment Complete!"
```

## 📞 Support

For any deployment issues:
- Create an issue on GitHub
- Contact AlgoRythm Tech support
- Check our documentation

---

**Built with ❤️ by AlgoRythm Tech**  
*The First Ever Teen-Built AI Startup*  
*CEO & Founder: Sri Aasrith Souri Kompella*  
*Hyderabad, India*
