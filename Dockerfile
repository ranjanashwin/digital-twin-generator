# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for face processing
RUN pip3 install --no-cache-dir \
    insightface==0.7.3 \
    onnxruntime-gpu==1.15.0 \
    facexlib==0.3.0 \
    basicsr==1.4.2 \
    gfpgan==1.3.8 \
    codeformer==0.1.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/ip_adapter models/sdxl models/face_enhance selfies output web

# Download models (this will take a while)
RUN python3 download_models.py --force

# Expose port
EXPOSE 5000

# Set default command
CMD ["python3", "app.py"] 