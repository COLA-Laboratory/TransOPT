#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DOCKER_ROOT_DIR="$SCRIPT_DIR/../resources/docker"

remove_old_images() {
    local image_name=$1

    old_image_ids=$(docker images -q --filter "reference=$image_name" | tail -n +2)
    
    if [ -n "$old_image_ids" ]; then
        echo "Removing old Docker image(s) with name '$image_name'..."
        docker rmi -f $old_image_ids
    fi
    
    dangling_image_ids=$(docker images -f "dangling=true" -q)
    if [ -n "$dangling_image_ids" ]; then
        echo "Removing dangling images..."
        docker rmi -f $dangling_image_ids
    fi
}

build_docker_image() {
    local image_name=$1
    local docker_dir=$2
    
    if [ -f "$docker_dir/Dockerfile" ]; then
        echo "Building Docker image '$image_name'..."
        docker build -t "$image_name" "$docker_dir"
        echo "Docker image '$image_name' created successfully."

        remove_old_images "$image_name"
    else
        echo "Dockerfile not found in $docker_dir"
        exit 1
    fi
}

# 构建 absolut_image
build_docker_image "absolut_image" "$DOCKER_ROOT_DIR/absolut_image"
