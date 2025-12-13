# Frontend Setup Guide

## Step 1: Install Node.js

If you don't have Node.js installed, install it first:

### Ubuntu/Debian:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### macOS (using Homebrew):
```bash
brew install node
```

### Verify installation:
```bash
node --version  # Should show v20.x.x or higher
npm --version   # Should show 10.x.x or higher
```

## Step 2: Install Dependencies

```bash
cd frontend
npm install
```

This will install:
- React 18
- Vite (build tool)
- Axios (HTTP client)

## Step 3: Start Development Server

```bash
npm run dev
```

The frontend will be available at: **http://localhost:3000**

## Step 4: Make Sure Backend is Running

The frontend needs the backend API to be running on port 8000:

```bash
# In another terminal, from the project root:
cd backend
python3.10 -m uvicorn main:app --reload --port 8000
```

## Step 5: Test the Chat

1. Open http://localhost:3000 in your browser
2. Type a query like: "I want spicy Thai food"
3. You should see restaurant recommendations!

## Troubleshooting

### Port 3000 already in use:
```bash
# Kill the process using port 3000
lsof -ti:3000 | xargs kill -9
```

### Backend connection error:
- Make sure backend is running on port 8000
- Check that CORS is enabled in backend
- Verify the API_URL in vite.config.js

### Dependencies installation fails:
```bash
# Clear npm cache and try again
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

