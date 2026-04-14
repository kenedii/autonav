# Dockerfile for Google Cloud Run (CPU Optimized)
FROM python:3.10-slim

# Install supervisord + nginx + curl
RUN apt-get update && apt-get install -y --no-install-recommends     supervisor curl nginx &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY gcp_deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app.py .
COPY tools ./tools
COPY model_api/resnet_api.py ./model_api/

# Copy deployment configs
COPY gcp_deploy/nginx.conf /etc/nginx/nginx.conf
COPY gcp_deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Final command
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
