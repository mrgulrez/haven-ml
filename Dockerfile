# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch for CPU (change for GPU)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Create directories
RUN mkdir -p logs data/profiles models/llm

# Expose port (if using web interface)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

# Run the agent
CMD ["python", "main.py"]
