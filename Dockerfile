# Use a slim image to keep size down
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (needed for some transformer/numpy libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install dependencies (Cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the local model specifically
# Ensure this directory structure matches your SentenceTransformer(MODEL_ID) call
# COPY ./krutrim-ai-labs/Vyakyarth ./krutrim-ai-labs/Vyakyarth

# 3. Copy application code
COPY . .

# 4. Create a .dockerignore to exclude .env, venv, and __pycache__
# (See the .dockerignore snippet below)

EXPOSE 8085

# Streamlit-specific healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8085/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8085", "--server.address=0.0.0.0"]