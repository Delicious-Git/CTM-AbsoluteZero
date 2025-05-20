FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/embeddings models logs state

# Set environment variables
ENV CTM_AZ_CONFIG="/app/configs/production.yaml"
ENV PYTHONPATH="/app"

# Expose ports
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.cli", "--config", "/app/configs/production.yaml", "dfz"]