#!/bin/bash

# run.sh - Script to build and run the Docker Compose application

# Variables
DOCKER_COMPOSE_FILE="docker/docker-compose.yml"

# Navigate to the directory containing the docker-compose.yml file
cd "$(dirname "$DOCKER_COMPOSE_FILE")"

echo "Building Docker images..."
docker compose -f "$(basename "$DOCKER_COMPOSE_FILE")" build

echo "Starting Docker containers..."
docker compose -f "$(basename "$DOCKER_COMPOSE_FILE")" up
