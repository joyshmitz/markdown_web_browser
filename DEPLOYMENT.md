# Markdown Web Browser - Production Deployment Guide

## Quick Start

### One-Line Installation
```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/markdown_web_browser/main/install.sh | bash -s -- --yes
```

### Deploy with Docker Compose
```bash
./deploy.sh --compose deploy
```

### Deploy to Kubernetes
```bash
./deploy.sh --env production deploy
```

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Deployment Options](#deployment-options)
- [Monitoring](#monitoring)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **CPU**: 2+ cores recommended (4+ for production)
- **Memory**: 4GB minimum (8GB+ recommended)
- **Storage**: 20GB+ for cache and artifacts
- **OS**: Ubuntu 20.04+, Debian 11+, RHEL 8+, or macOS

### Software Dependencies
- Docker 20.10+ or Podman
- Docker Compose v2+ (for compose deployment)
- Kubernetes 1.24+ (for k8s deployment)
- libvips 8.10+ (installed automatically)

## Installation Methods

### 1. Automated Installation (Recommended)

The installer script handles all dependencies:

```bash
# Interactive installation
curl -fsSL https://raw.githubusercontent.com/yourusername/markdown_web_browser/main/install.sh | bash

# Non-interactive with custom directory
curl -fsSL https://raw.githubusercontent.com/yourusername/markdown_web_browser/main/install.sh | bash -s -- \
  --yes --dir=/opt/mdwb --ocr-key=your-api-key
```

### 2. Docker Installation

#### Build from source:
```bash
docker build -t markdown-web-browser:latest .
```

#### Or pull from registry:
```bash
docker pull yourusername/markdown-web-browser:latest
```

#### Run container:
```bash
docker run -d \
  --name mdwb \
  -p 8000:8000 \
  -p 9000:9000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.cache:/app/.cache \
  -e OLMOCR_API_KEY=your-api-key \
  markdown-web-browser:latest
```

### 3. Docker Compose Deployment

#### Development:
```bash
docker-compose --profile dev up -d
```

#### Production with monitoring:
```bash
docker-compose --profile production --profile monitoring up -d
```

### 4. Kubernetes Deployment

#### Apply manifests:
```bash
kubectl apply -f k8s/production/
```

#### Using Helm (if available):
```bash
helm install mdwb ./helm/markdown-web-browser \
  --namespace mdwb \
  --create-namespace \
  --values helm/values-production.yaml
```

## Configuration

### Environment Variables

Create a `.env` file from the example:
```bash
cp .env.example .env
```

**Required variables:**
```env
# OCR API Configuration
OLMOCR_SERVER=https://ai2endpoints.cirrascale.ai/api
OLMOCR_API_KEY=your-api-key-here
OLMOCR_MODEL=olmOCR-2-7B-1025-FP8

# API Configuration
API_BASE_URL=http://your-domain.com:8000
MDWB_API_KEY=your-internal-api-key
```

**Production optimizations:**
```env
# Server Configuration
MDWB_SERVER_IMPL=granian  # Use Granian for production
MDWB_SERVER_WORKERS=4      # Number of worker processes
MDWB_GRANIAN_RUNTIME_THREADS=2  # Threads per worker

# Performance Tuning
OCR_MAX_CONCURRENCY=8      # Max parallel OCR requests
OCR_MIN_CONCURRENCY=2      # Min parallel OCR requests
MAX_VIEWPORT_SWEEPS=200    # Max screenshots per page
SCROLL_SETTLE_MS=350       # Time to wait after scrolling

# Cache Configuration
CACHE_ROOT=/app/.cache     # Cache directory
```

### SSL/TLS Configuration

For HTTPS support, use a reverse proxy:

#### Nginx example:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Deployment Options

### Small Scale (1-10 users)
- Single Docker container
- SQLite database
- Local cache storage
- 2 CPU cores, 4GB RAM

```bash
./deploy.sh --compose --env staging deploy
```

### Medium Scale (10-100 users)
- Docker Compose with multiple workers
- Redis for job queue
- Shared cache volume
- 4-8 CPU cores, 8-16GB RAM

```bash
docker-compose --profile production up -d --scale web=3
```

### Large Scale (100+ users)
- Kubernetes deployment
- PostgreSQL database
- Redis cluster
- S3-compatible object storage
- Horizontal pod autoscaling

```bash
./deploy.sh --env production --namespace mdwb deploy
```

## Monitoring

### Prometheus Metrics

Metrics exposed on port 9000:
- Request latency
- OCR processing time
- Cache hit rates
- Error rates
- Concurrent jobs

### Health Checks

- **Liveness**: `GET /health`
- **Readiness**: `GET /ready`
- **Metrics**: `GET /metrics`

### Logging

Configure structured logging:
```env
MDWB_SERVER_LOG_LEVEL=info
LOG_FORMAT=json
```

View logs:
```bash
# Docker
docker logs -f mdwb

# Docker Compose
docker-compose logs -f web

# Kubernetes
kubectl logs -f deployment/markdown-web-browser
```

## Security

### API Authentication
```python
# Enable API key authentication
MDWB_API_KEY=generate-strong-key-here
```

### Network Security
- Use HTTPS in production
- Implement rate limiting
- Configure CORS properly
- Use firewall rules

### Container Security
- Run as non-root user
- Use read-only filesystem where possible
- Scan images for vulnerabilities
- Keep dependencies updated

## Troubleshooting

### Common Issues

#### 1. PNG Encoding Errors
```
Error: unable to call VipsForeignSaveSpngTarget
```
**Solution**: Update libvips or restart the service

#### 2. OCR Connection Failed
```
Error: olmOCR request failed after 3 attempts
```
**Solution**: Check API credentials and network connectivity

#### 3. Playwright Browser Issues
```
Error: Executable doesn't exist at /home/ubuntu/.cache/ms-playwright
```
**Solution**: Run `playwright install chromium`

#### 4. High Memory Usage
**Solution**: Adjust worker count and concurrency settings:
```env
MDWB_SERVER_WORKERS=2
OCR_MAX_CONCURRENCY=4
```

### Performance Tuning

#### For faster captures:
```env
SCROLL_SETTLE_MS=200      # Reduce scroll wait time
MAX_VIEWPORT_SWEEPS=100   # Limit screenshots
OCR_MAX_BATCH_TILES=5     # Larger OCR batches
```

#### For better quality:
```env
SCROLL_SETTLE_MS=500      # More time for content to load
DEVICE_SCALE_FACTOR=3     # Higher resolution
CAPTURE_LONG_SIDE_PX=1920 # Larger tiles
```

## Maintenance

### Backup
```bash
# Backup data and cache
tar -czf backup-$(date +%Y%m%d).tar.gz data/ .cache/ ops/

# Backup database (if using PostgreSQL)
pg_dump markdown_web_browser > backup.sql
```

### Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and deploy
./deploy.sh --compose update
```

### Cleanup
```bash
# Remove old cache files (older than 30 days)
find .cache -type f -mtime +30 -delete

# Clean Docker resources
docker system prune -af
```

## Support

- **Documentation**: See `/docs` directory
- **Issues**: GitHub Issues
- **Logs**: Check `/ops/warnings.jsonl` for capture warnings
- **Metrics**: Monitor Prometheus endpoint at `:9000/metrics`

## License

See LICENSE file for details.