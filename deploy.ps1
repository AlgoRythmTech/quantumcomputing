# RythmAI Europa Deployment Script for Windows
# Created by AlgoRythm Tech - CEO: Sri Aasrith Souri Kompella

Write-Host "🚀 RythmAI Europa Deployment Script" -ForegroundColor Cyan
Write-Host "Created by AlgoRythm Tech - The First Ever Teen-Built AI Startup" -ForegroundColor Green
Write-Host "CEO & Founder: Sri Aasrith Souri Kompella" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Menu for deployment options
Write-Host "`nSelect deployment option:" -ForegroundColor Yellow
Write-Host "1. Deploy Frontend to Vercel" -ForegroundColor White
Write-Host "2. Deploy Frontend to Netlify" -ForegroundColor White
Write-Host "3. Deploy Backend to Render" -ForegroundColor White
Write-Host "4. Deploy Both (Frontend + Backend)" -ForegroundColor White
Write-Host "5. Run Locally (Development)" -ForegroundColor White
Write-Host "6. Exit" -ForegroundColor White

$choice = Read-Host "`nEnter your choice (1-6)"

switch ($choice) {
    "1" {
        Write-Host "`n📦 Deploying Frontend to Vercel..." -ForegroundColor Cyan
        Set-Location -Path "frontend"
        
        # Check if Vercel CLI is installed
        if (-not (Test-Command "vercel")) {
            Write-Host "Installing Vercel CLI..." -ForegroundColor Yellow
            npm.cmd install -g vercel
        }
        
        # Build the frontend
        Write-Host "Building frontend..." -ForegroundColor Yellow
        npm.cmd run build
        
        # Deploy to Vercel
        Write-Host "Deploying to Vercel..." -ForegroundColor Yellow
        vercel --prod
        
        Write-Host "✅ Frontend deployed to Vercel!" -ForegroundColor Green
    }
    
    "2" {
        Write-Host "`n📦 Deploying Frontend to Netlify..." -ForegroundColor Cyan
        Set-Location -Path "frontend"
        
        # Check if Netlify CLI is installed
        if (-not (Test-Command "netlify")) {
            Write-Host "Installing Netlify CLI..." -ForegroundColor Yellow
            npm.cmd install -g netlify-cli
        }
        
        # Build the frontend
        Write-Host "Building frontend..." -ForegroundColor Yellow
        npm.cmd run build
        
        # Deploy to Netlify
        Write-Host "Deploying to Netlify..." -ForegroundColor Yellow
        netlify deploy --prod --dir=dist
        
        Write-Host "✅ Frontend deployed to Netlify!" -ForegroundColor Green
    }
    
    "3" {
        Write-Host "`n🔧 Backend Deployment to Render" -ForegroundColor Cyan
        Write-Host "Please follow these steps:" -ForegroundColor Yellow
        Write-Host "1. Push your code to GitHub" -ForegroundColor White
        Write-Host "2. Go to https://render.com" -ForegroundColor White
        Write-Host "3. Create a new Web Service" -ForegroundColor White
        Write-Host "4. Connect your GitHub repository" -ForegroundColor White
        Write-Host "5. Use these settings:" -ForegroundColor White
        Write-Host "   - Build Command: pip install -r requirements.txt" -ForegroundColor Gray
        Write-Host "   - Start Command: uvicorn algorythm_ai_backend:app --host 0.0.0.0 --port `$PORT" -ForegroundColor Gray
        Write-Host "6. Deploy!" -ForegroundColor White
        
        $openBrowser = Read-Host "`nOpen Render.com in browser? (y/n)"
        if ($openBrowser -eq "y") {
            Start-Process "https://render.com"
        }
    }
    
    "4" {
        Write-Host "`n🚀 Deploying Both Frontend and Backend..." -ForegroundColor Cyan
        
        # Deploy Frontend
        Write-Host "`nStep 1: Frontend Deployment" -ForegroundColor Yellow
        Set-Location -Path "frontend"
        npm.cmd run build
        
        $frontendChoice = Read-Host "Deploy frontend to (1) Vercel or (2) Netlify?"
        if ($frontendChoice -eq "1") {
            if (-not (Test-Command "vercel")) {
                npm.cmd install -g vercel
            }
            vercel --prod
        } else {
            if (-not (Test-Command "netlify")) {
                npm.cmd install -g netlify-cli
            }
            netlify deploy --prod --dir=dist
        }
        
        # Backend instructions
        Write-Host "`nStep 2: Backend Deployment" -ForegroundColor Yellow
        Write-Host "Push your code to GitHub and deploy via Render.com" -ForegroundColor White
        Start-Process "https://render.com"
        
        Write-Host "`n✅ Deployment process initiated!" -ForegroundColor Green
    }
    
    "5" {
        Write-Host "`n🖥️ Running Locally..." -ForegroundColor Cyan
        
        # Start backend
        Write-Host "Starting backend server..." -ForegroundColor Yellow
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python algorythm_ai_backend.py"
        
        Start-Sleep -Seconds 3
        
        # Start frontend
        Write-Host "Starting frontend server..." -ForegroundColor Yellow
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm.cmd run dev"
        
        Write-Host "`n✅ Local development servers started!" -ForegroundColor Green
        Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
        Write-Host "Backend: http://localhost:8000" -ForegroundColor White
    }
    
    "6" {
        Write-Host "`nGoodbye! Thank you for using RythmAI Europa" -ForegroundColor Cyan
        exit
    }
    
    default {
        Write-Host "`n❌ Invalid choice. Please run the script again." -ForegroundColor Red
    }
}

Write-Host "`n" + "=" * 60 -ForegroundColor Gray
Write-Host "Thank you for using RythmAI Europa!" -ForegroundColor Cyan
Write-Host "Created with ❤️ by AlgoRythm Tech" -ForegroundColor Green
