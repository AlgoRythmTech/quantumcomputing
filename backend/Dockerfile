# AlgoRythm AI Europa Backend Docker Configuration
# Created by AlgoRythm Tech - CEO: Sri Aasrith Souri Kompella

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs checkpoints

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "algorythm_ai_backend:app", "--host", "0.0.0.0", "--port", "8000"]
