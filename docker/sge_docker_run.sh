#!/bin/bash

# Function to convert host paths to container paths and build docker mounts
process_args() {
    container_args=()
    docker_mounts=()
    gpu_index="all"  # Default to all GPUs
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gpu)
                gpu_index="$2"
                shift 2
                ;;
            --out)
                # For --out argument, mount the parent directory
                out_dir=$(dirname "$2")
                out_file=$(basename "$2")
                # Convert to absolute path
                abs_path=$(realpath "$out_dir")
                # Create a hash of the absolute path for unique mount point
                path_hash=$(echo -n "$abs_path" | md5sum | cut -c1-8)
                # Create mount point with hash to avoid collisions
                mount_point="/data/${path_hash}_$(basename "$abs_path")"
                # Add to docker mounts with write permissions
                docker_mounts+=("-v" "${abs_path}:${mount_point}:rw")
                # Create directory and set permissions
                mkdir -p "$abs_path"
                # Replace the argument with the container path
                container_args+=("$1" "${mount_point}/${out_file}")
                shift 2
                ;;
            *)
                # Check if argument is a file or directory path
                if [ -e "$1" ]; then
                    # Convert to absolute path
                    abs_path=$(realpath "$1")
                    # Create a hash of the absolute path for unique mount point
                    path_hash=$(echo -n "$abs_path" | md5sum | cut -c1-8)
                    # Create mount point with hash to avoid collisions
                    mount_point="/data/${path_hash}_$(basename "$abs_path")"
                    # Add to docker mounts
                    docker_mounts+=("-v" "${abs_path}:${mount_point}")
                    # Replace the argument with the container path
                    container_args+=("$mount_point")
                else
                    # Pass through non-path arguments
                    container_args+=("$1")
                fi
                shift
                ;;
        esac
    done

    # Export gpu_index for use in main script
    export gpu_index
}

# Check if the cache volume exists, create if it doesn't
if ! docker volume inspect sge_models_cache >/dev/null 2>&1; then
    echo "Creating sge_models_cache volume..."
    docker volume create sge_models_cache
    # Set permissions on the volume
    docker run --rm -v sge_models_cache:/data busybox chown -R 9001:9001 /data
fi

# Process arguments
process_args "$@"

# Check which image exists and use the appropriate one
if [[ "$(docker images -q speech_gen_eval:latest 2> /dev/null)" != "" ]]; then
    image_name="speech_gen_eval"
elif [[ "$(docker images -q balacoon/speech_gen_eval:0.1 2> /dev/null)" != "" ]]; then
    image_name="balacoon/speech_gen_eval:0.1"
else
    echo "Error: Neither speech_gen_eval nor balacoon/speech_gen_eval image found"
    echo "Please build the image with docker/build.sh or pull from Docker Hub"
    exit 1
fi

# Run the container with proper mounts and GPU support
docker run --rm \
    --gpus "device=$gpu_index" \
    -e LOCAL_USER_ID="$(id -u)" \
    -e LOCAL_GROUP_ID="$(id -g)" \
    -v sge_models_cache:/home/appuser/.cache:rw \
    "${docker_mounts[@]}" \
    "$image_name" "${container_args[@]}"
