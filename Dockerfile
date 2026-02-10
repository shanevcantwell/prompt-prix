FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY prompt_prix/ prompt_prix/

# Install the package
RUN pip install --no-cache-dir -e .

# Gradio port
EXPOSE 7860

# Environment defaults (can be overridden)
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1

# Run the app
CMD ["prompt-prix"]
