# 🍽️ Swaad - Flavor Profile Recipe Recommendation System

A full-stack application that creates personalized flavor profiles based on your favorite dishes and recommends menu items that match your taste preferences.

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.7+** (Python 3.8 or higher recommended)
  - Check your version: `python3 --version` or `python --version`
  - Download from [python.org](https://www.python.org/downloads/) if needed
- **Node.js 16+** and **npm** (Node Package Manager)
  - Check your version: `node --version` and `npm --version`
  - Download from [nodejs.org](https://nodejs.org/) if needed
- **Git** (for cloning the repository)
  - Check your version: `git --version`

### Required Files

Make sure the following file exists in the project root directory:
- `recipes_with_flavour_profiles.csv` - Recipe database with flavor profiles

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd swaad
```

### Step 2: Set Up Your API Key

1. Open the `sample_env_file.env` file in the project root folder
2. Replace `your_api_key_here` with your actual Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
3. Rename the file from `sample_env_file.env` to `.env`

**How to get a Gemini API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **"Get API Key"** in the sidebar
4. Click **"Create API key"** → Select **"Create API key in new project"**
5. Copy your API key (it starts with `AIza...`)

---

## ▶️ Starting the Application (Demo)

### 🚀 Method 1: Automated Script (Recommended)

Run the start script which will automatically create a virtual environment and install all dependencies:

```bash
./start.sh
```

Once complete, open your browser and go to **http://localhost:3000** — you're all set! 🎉

---

### 🔧 Method 2: Manual Installation

If the script doesn't work or you prefer manual setup, run these commands from the project root:

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install backend dependencies:**
   ```bash
   pip install -r ./backend/requirements.txt
   ```

3. **Start the backend server:**
   ```bash
   python ./backend/main.py
   ```

4. **Open a new terminal, activate venv, and install frontend dependencies:**
   ```bash
   source venv/bin/activate
   npm install --prefix ./frontend
   ```

5. **Start the frontend server:**
   ```bash
   npm run dev --prefix ./frontend
   ```

6. **Open your browser and go to** **http://localhost:3000** — done! 🎉

---

## 📖 Usage Guide

### Creating Your Flavor Profile

1. **Enter Favorite Dishes:**
   - Navigate to the profile creation page
   - Enter dishes you like in three categories:
     - **Appetizers/Starters**
     - **Main Courses/Mains**
     - **Desserts**
   - Use the search feature to find dishes from the recipe database
   - You can add multiple dishes in each category

2. **Generate Profile:**
   - Click "Create My Flavor Profile"
   - The system calculates your flavor preferences based on the ingredients in your favorite dishes
   - You'll see a visual radar chart showing your flavor profile

3. **View Your Profile:**
   - See detailed flavor percentages for each category
   - Compare profiles across different meal types
   - Your profile is saved for the current session

### Getting Recommendations

1. **Upload or Enter Menu:**
   - **Option A:** Paste menu text directly into the text area
   - **Option B:** Type dish names (one per line)
   - **Option C:** Upload a menu image (uses Gemini API to extract dish names automatically)

2. **Process Menu:**
   - The system automatically extracts dish names from the text
   - It recognizes category headers (appetizers, mains, desserts, etc.)
   - Dishes are matched against the recipe database

3. **View Recommendations:**
   - Click "Get Recommendations"
   - Dishes are sorted by flavor similarity to your profile
   - Each recommendation shows:
     - Match percentage
     - Flavor profile comparison
     - Ingredients list (expandable)
   - Higher match percentages indicate better alignment with your preferences

---

## 🔧 Troubleshooting

### General Issues

**Problem: Changes not reflecting**
- **Solution:** Both servers support hot-reload. If changes don't appear:
  - Save your files
  - Check terminal for errors
  - Hard refresh browser (Ctrl+Shift+R or Cmd+Shift+R)

**Problem: Menu image upload not working**
- **Solution 1:** Verify Gemini API key is set:
  ```bash
  cat .env | grep GEMINI_API_KEY
  ```
  If not set, follow the setup instructions in the Installation section.

- **Solution 2:** Verify google-genai is installed:
  ```bash
  pip install google-genai
  ```

- **Solution 3:** Check API key is valid:
  - Make sure the API key starts with `AIza...`
  - Verify the key is active at [Google AI Studio](https://aistudio.google.com)
  - Ensure there are no extra spaces or quotes in the `.env` file

---

## 👨‍💻 Author

**Bhanu Sharma**  
sharma.bhan@northeastern.edu

---

Enjoy using Swaad! 🍽️


