# NeerSetu Deployment Guide

## 🚀 Environment Setup

### Local Development

1. **Frontend Environment Setup:**
   ```bash
   # Copy the example environment file
   cp env.example .env.local
   
   # Edit .env.local with your settings
   NEXT_PUBLIC_API_URL=http://localhost:8000
   NODE_ENV=development
   ```

2. **Backend Environment Setup:**
   ```bash
   cd backend_tisha
   
   # Copy the example environment file
   cp env.example env
   
   # Edit env with your settings
   GROQ_API_KEY=your_groq_api_key_here
   PORT=8000
   NODE_ENV=development
   ```

### Production Deployment

#### Option 1: Vercel (Frontend) + Render (Backend)

**Frontend (Vercel):**
1. Deploy to Vercel
2. Set environment variables in Vercel dashboard:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
   NODE_ENV=production
   ```

**Backend (Render):**
1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   PORT=10000
   NODE_ENV=production
   FRONTEND_URL=https://your-frontend-url.vercel.app
   ```

#### Option 2: Both on Render

Use the `render.yaml` configuration file for automatic deployment.

## 🔧 Environment Variables

### Frontend (.env.local)
```env
# Local Development
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development

# Production
# NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
# NODE_ENV=production
```

### Backend (env)
```env
# Local Development
GROQ_API_KEY=your_groq_api_key_here
PORT=8000
NODE_ENV=development

# Production
# GROQ_API_KEY=your_groq_api_key_here
# PORT=10000
# NODE_ENV=production
# FRONTEND_URL=https://your-frontend-url.vercel.app
```

## 🚀 Quick Start Commands

### Local Development
```bash
# Start Backend
cd backend_tisha
pip install -r requirements.txt
python app.py

# Start Frontend (in another terminal)
npm install
npm run dev
```

### Production URLs
- **Frontend:** https://your-app.vercel.app
- **Backend:** https://your-app.onrender.com

## 📝 Notes

1. **CORS Configuration:** The backend automatically allows requests from localhost and production URLs
2. **Environment Detection:** The app automatically detects the environment and uses appropriate URLs
3. **API Key Security:** Never commit your GROQ_API_KEY to version control
4. **Port Configuration:** Render uses port 10000, local development uses 8000
