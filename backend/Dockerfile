# Use Python 3.11 as specified in runtime.txt
# 
# Build instructions:
# Build from the parent directory (Swaad/) with:
#   docker build -f backend/Dockerfile .
# This allows access to both backend files and the recipes CSV file.
#
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# gcc, g++ for compiling Python packages
# git for sentence-transformers model downloads
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend Python files (excluding test files and db_scripts)
COPY backend/*.py .

# Copy routes directory
COPY backend/routes ./routes

# Copy required data files
COPY backend/ingredient-flavor.csv ./ingredient-flavor.csv

# Copy the recipes CSV file (if it exists in parent directory)
# The application looks for it in current directory or parent directory
COPY recipes_with_flavour_profiles.csv ./recipes_with_flavour_profiles.csv

# Create directory for database (SQLite will create the file if it doesn't exist)
RUN mkdir -p /app/data

# Expose the port that the app runs on
EXPOSE 8000

# Environment variables are loaded from .env file at runtime using --env-file
# See .env.example for required and optional variables
# Run with: docker run --env-file .env -p 8000:8000 swaad-backend

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

