# Use an official Python 3.10 runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COQUI_MODEL_PATH=/app/models
ENV DEVICE=cpu
ENV COMPUTE_TYPE=int8

# Install system dependencies
# We need gcc/g++ for compiling some Python extensions
# We need ffmpeg and its development headers for the 'av' package
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    python3-dev \
    pkg-config \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavdevice-dev \
    libavfilter-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
# Update pip first for better performance
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Run the application with optimized settings for AI workloads
# --workers 1 is essential to avoid loading the 1.77GB model multiple times
# --timeout-keep-alive 300 allows time for model initialization
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "300"]
