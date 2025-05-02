FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m GNN

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/model /app/utils /app/logs \
    && chown -R GNN:GNN /app

# Copy project files
COPY . .
RUN chown -R GNN:GNN /app

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=-1  # Use CPU mode by default
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1  # Force Python to run in unbuffered mode

# Create volumes for persistent data
VOLUME ["/app/data", "/app/model", "/app/logs"]

# Expose port for potential web interface or API
EXPOSE 8000

# Switch to non-root user
USER GNN

# Replace complex healthcheck with a simple one
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

# Create a startup script for better debugging
RUN echo '#!/bin/bash\necho "Starting GNN application..."\nls -la\nif [ -f "main.py" ]; then\n  echo "Running main.py"\n  python -u main.py\nelse\n  echo "Error: main.py not found!"\n  echo "Contents of current directory:"\n  ls -la\n  python -c "import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")" \n  /bin/bash\nfi' > /app/start.sh && \
    chmod +x /app/start.sh

# Run the startup script instead of directly running main.py
CMD ["/app/start.sh"]
