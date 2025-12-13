#!/bin/bash

echo "🍽️  Starting Swaad - AI Restaurant Recommendations"
echo "=================================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check Python
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 not found!"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found! Please install Node.js 18+"
    echo "   Ubuntu: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
    exit 1
fi

echo "✅ Python: $(python3.10 --version)"
echo "✅ Node.js: $(node --version)"
echo ""

# Start Backend
echo "🔧 Starting Backend (Port 8000)..."
cd backend
python3.10 -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..
sleep 3

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start Frontend
echo "🎨 Starting Frontend (Port 3000)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Servers started successfully!"
echo ""
echo "📍 Backend API:  http://localhost:8000"
echo "📍 Frontend App: http://localhost:3000"
echo "📍 API Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for processes
wait

