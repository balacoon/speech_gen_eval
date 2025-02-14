# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies including Python and ffmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies globally
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN GIT_LFS_SKIP_SMUDGE=1 pip3 install --no-cache-dir -r requirements.txt

# Copy the entire repository
COPY . .

# Install the package
RUN pip3 install .

# Copy entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
