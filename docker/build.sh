#!/bin/bash

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build the Docker image
docker build \
    -t speech_gen_eval \
    -f docker/Dockerfile \
    .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Docker image 'speech_gen_eval' built successfully!"
else
    echo "Docker build failed!"
    exit 1
fi
