# Swaad Frontend

Modern React-based chat interface for AI-powered restaurant recommendations.

## Features

- 🎨 Beautiful, modern UI with gradient design
- 💬 Real-time chat interface
- 🍽️ Restaurant and dish recommendations
- 📱 Responsive design
- ⚡ Fast and lightweight (Vite + React)

## Prerequisites

- Node.js 18+ and npm

## Installation

```bash
cd frontend
npm install
```

## Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Environment Variables

Create a `.env` file in the frontend directory:

```
VITE_API_URL=http://localhost:8000
```

## Build for Production

```bash
npm run build
```

The build output will be in the `dist` directory.

## API Integration

The frontend connects to the backend API at `/api/chat` endpoint:

**Request:**
```json
{
  "query": "I want spicy Thai food"
}
```

**Response:**
```json
{
  "response": {
    "text": "Here are some recommendations..."
  },
  "menu_buddy": {
    "recommendations": [
      {
        "name": "Restaurant Name",
        "rating": 4.5,
        "price_range": 2,
        "cuisine_types": ["Thai"],
        "recommended_dishes": [
          {
            "name": "Pad Thai",
            "similarity": 0.85
          }
        ]
      }
    ]
  }
}
```

## Tech Stack

- React 18
- Vite
- Axios
- CSS3 (with animations)

