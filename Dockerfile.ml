# ML Training Service Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements_db.txt .
COPY EDA/requirements.txt ./eda_requirements.txt
COPY prediction/models/requirements.txt ./prediction_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_db.txt && \
    pip install --no-cache-dir -r eda_requirements.txt && \
    pip install --no-cache-dir -r prediction_requirements.txt && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    wandb \
    tqdm \
    matplotlib \
    numpy \
    pandas \
    jupyter \
    psycopg2-binary \
    python-dotenv

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for potential API
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "cbow.py"] 