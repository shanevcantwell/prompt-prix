#!/bin/bash
# Run tests inside Docker container
# Usage: ./scripts/docker-test.sh [pytest args...]

set -e

# Build image first
docker compose build --quiet

# Run tests with mounted volumes
docker compose run --rm \
  -v "$(pwd)/tests:/app/tests" \
  -v "$(pwd)/examples:/app/examples" \
  prompt-prix sh -c "pip install pytest pytest-asyncio respx -q && python -m pytest tests/ $*"
