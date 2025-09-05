# -----------------------------
# Base image
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y \
        poppler-utils \
        build-essential \
        libssl-dev \
        libffi-dev \
        libjpeg-dev \
        zlib1g-dev \
        git \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy requirements
# -----------------------------
COPY requirements.txt .

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy app code
# -----------------------------
COPY . .

# -----------------------------
# Expose port for Railway
# -----------------------------
EXPOSE 8000

# -----------------------------
# Command to run the FastAPI app
# -----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
