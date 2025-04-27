# Stage 1: Model Downloader
FROM python:3.10-slim AS model-downloader
LABEL stage=model-downloader

# Install huggingface-cli
RUN pip install --no-cache-dir huggingface_hub

# Set working directory
WORKDIR /model-downloader

# Create directory for downloaded Dia model
RUN mkdir -p /model-downloader/models/dia-1.6b

# Accept Hugging Face token at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Login with token if provided (useful for gated models, might not be strictly needed for Dia-1.6B)
RUN if [ -n "$HF_TOKEN" ]; then \
    echo "Logging in to Hugging Face Hub..."; \
    huggingface-cli login --token ${HF_TOKEN}; \
    else echo "Skipping Hugging Face Hub login (no HF_TOKEN provided)."; fi

# Download Dia-1.6B model files (config and checkpoint)
# No need for TTS_ENGINE check here, as this project *only* uses Dia
RUN echo "Downloading Dia-1.6B model files..." && \
    huggingface-cli download nari-labs/Dia-1.6B config.json --local-dir /model-downloader/models/dia-1.6b --local-dir-use-symlinks False && \
    huggingface-cli download nari-labs/Dia-1.6B dia-v0_1.pth --local-dir /model-downloader/models/dia-1.6b --local-dir-use-symlinks False && \
    echo "Dia-1.6B model files downloaded."

# Optional: Download DAC model here too to avoid runtime download?
# RUN echo "Downloading DAC model..." && \
#    huggingface-cli download descript/dac descript-dac-44khz/pytorch_model.bin --local-dir /model-downloader/models/dac --local-dir-use-symlinks False && \
#    huggingface-cli download descript/dac descript-dac-44khz/config.json --local-dir /model-downloader/models/dac --local-dir-use-symlinks False && \
#    echo "DAC model downloaded."
# DAC Download requires knowing the exact files/structure, adjust above ^^ if implemented.


# Stage 2: Main Application
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS runner
LABEL stage=runner

# Set environment variables for Python and CUDA/Torch
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    # Set Path for python libs
    # Needed for finding libraries installed via pip
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies required by the application and libs
# Combined dependencies from both original Dockerfiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create directories needed at runtime *before* installing requirements
# Models dir will hold models from builder stage, cache dir for runtime downloads (like DAC if not built-in)
RUN mkdir -p /app/voices /app/cache /app/static/audio /app/models/dia-1.6b \
    && chmod -R 777 /app/voices /app/cache /app/static

# Install Python dependencies
# Upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip
# Install specific versions or ranges if dia/dac require them
# Base requirements
RUN pip3 install --no-cache-dir -r requirements.txt
# Install dia library itself from git (needed for imports)
# Ensure dependencies match Dia's requirements (check its pyproject.toml if errors occur)
RUN pip3 install --no-cache-dir "git+https://github.com/nari-labs/dia.git"

# Copy application code
COPY ./app /app/app
# Copy static files and potentially other root-level files if needed
COPY ./static /app/static

# Copy downloaded models from the model-downloader stage
COPY --from=model-downloader /model-downloader/models/dia-1.6b /app/models/dia-1.6b
# Optional: Copy DAC model if downloaded in builder stage
# COPY --from=model-downloader /model-downloader/models/dac /app/models/dac

# Ensure models are readable
RUN chmod -R 755 /app/models

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["python3", "-m", "app.main"]