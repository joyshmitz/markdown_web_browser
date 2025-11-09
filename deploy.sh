#!/usr/bin/env bash
# Production Deployment Script for Markdown Web Browser

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
ACTION="deploy"
DOCKER_REGISTRY=""
IMAGE_TAG="latest"
NAMESPACE="default"
DRY_RUN=false

# Print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print usage
usage() {
    cat << EOF
Markdown Web Browser - Deployment Script

Usage: $0 [OPTIONS] ACTION

Actions:
    deploy          Deploy the application
    update          Update existing deployment
    rollback        Rollback to previous version
    status          Show deployment status
    logs            Show application logs
    stop            Stop the deployment
    clean           Clean up resources

Options:
    --env ENV           Environment (development|staging|production) [default: production]
    --tag TAG           Docker image tag [default: latest]
    --registry REG      Docker registry URL
    --namespace NS      Kubernetes namespace [default: default]
    --compose           Use Docker Compose instead of Kubernetes
    --dry-run           Show what would be done without doing it
    --help, -h          Show this help message

Examples:
    # Deploy to production with Docker Compose
    $0 --compose deploy

    # Deploy to Kubernetes
    $0 --env production deploy

    # Update with specific tag
    $0 --tag v1.2.3 update

    # Check status
    $0 status

EOF
    exit 0
}

# Parse arguments
USE_COMPOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --compose)
            USE_COMPOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        deploy|update|rollback|status|logs|stop|clean)
            ACTION="$1"
            shift
            ;;
        *)
            print_color "$RED" "Unknown option: $1"
            usage
            ;;
    esac
done

# Build Docker image
build_image() {
    print_color "$BLUE" "Building Docker image..."

    local image_name="markdown-web-browser:${IMAGE_TAG}"

    if [ ! -z "$DOCKER_REGISTRY" ]; then
        image_name="${DOCKER_REGISTRY}/markdown-web-browser:${IMAGE_TAG}"
    fi

    if [ "$DRY_RUN" = true ]; then
        print_color "$YELLOW" "DRY RUN: Would build image $image_name"
        return 0
    fi

    docker build -t "$image_name" .

    if [ ! -z "$DOCKER_REGISTRY" ]; then
        print_color "$BLUE" "Pushing image to registry..."
        docker push "$image_name"
    fi

    print_color "$GREEN" "✓ Image built: $image_name"
}

# Deploy with Docker Compose
deploy_compose() {
    print_color "$BLUE" "Deploying with Docker Compose..."

    local compose_file="docker-compose.yml"
    local profiles=""

    case $ENVIRONMENT in
        development)
            profiles="--profile dev"
            ;;
        staging)
            profiles=""
            ;;
        production)
            profiles="--profile production --profile monitoring"
            ;;
    esac

    if [ "$DRY_RUN" = true ]; then
        print_color "$YELLOW" "DRY RUN: Would run: docker-compose $profiles up -d"
        return 0
    fi

    # Check if .env file exists
    if [ ! -f .env ]; then
        print_color "$YELLOW" "Creating .env from .env.example..."
        cp .env.example .env
        print_color "$YELLOW" "Please edit .env with your configuration"
    fi

    # Build and start services
    docker-compose $profiles build
    docker-compose $profiles up -d

    print_color "$GREEN" "✓ Services deployed with Docker Compose"

    # Show status
    docker-compose ps
}

# Deploy with Kubernetes
deploy_kubernetes() {
    print_color "$BLUE" "Deploying to Kubernetes..."

    if ! command -v kubectl &> /dev/null; then
        print_color "$RED" "kubectl not found. Please install kubectl first."
        exit 1
    fi

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if [ "$DRY_RUN" = true ]; then
            print_color "$YELLOW" "DRY RUN: Would create namespace $NAMESPACE"
        else
            print_color "$BLUE" "Creating namespace $NAMESPACE..."
            kubectl create namespace "$NAMESPACE"
        fi
    fi

    # Apply configurations
    local k8s_dir="k8s/${ENVIRONMENT}"

    if [ ! -d "$k8s_dir" ]; then
        print_color "$RED" "Kubernetes manifests not found in $k8s_dir"
        print_color "$YELLOW" "Creating basic manifests..."
        mkdir -p "$k8s_dir"
        create_k8s_manifests "$k8s_dir"
    fi

    if [ "$DRY_RUN" = true ]; then
        print_color "$YELLOW" "DRY RUN: Would apply manifests from $k8s_dir"
        kubectl apply -f "$k8s_dir" --dry-run=client -n "$NAMESPACE"
    else
        kubectl apply -f "$k8s_dir" -n "$NAMESPACE"
        print_color "$GREEN" "✓ Deployed to Kubernetes namespace: $NAMESPACE"
    fi
}

# Create basic Kubernetes manifests
create_k8s_manifests() {
    local dir="$1"

    cat > "$dir/deployment.yaml" << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: markdown-web-browser
  labels:
    app: mdwb
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mdwb
  template:
    metadata:
      labels:
        app: mdwb
    spec:
      containers:
      - name: web
        image: markdown-web-browser:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9000
          name: metrics
        env:
        - name: MDWB_SERVER_IMPL
          value: "granian"
        - name: MDWB_SERVER_WORKERS
          value: "4"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
EOF

    cat > "$dir/service.yaml" << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: markdown-web-browser
spec:
  selector:
    app: mdwb
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9000
    targetPort: 9000
  type: LoadBalancer
EOF

    print_color "$GREEN" "✓ Created Kubernetes manifests in $dir"
}

# Show status
show_status() {
    if [ "$USE_COMPOSE" = true ]; then
        print_color "$BLUE" "Docker Compose Status:"
        docker-compose ps
    else
        print_color "$BLUE" "Kubernetes Status:"
        kubectl get all -n "$NAMESPACE" -l app=mdwb
    fi
}

# Show logs
show_logs() {
    if [ "$USE_COMPOSE" = true ]; then
        docker-compose logs -f --tail=100
    else
        kubectl logs -f -n "$NAMESPACE" -l app=mdwb --max-log-requests=10
    fi
}

# Stop deployment
stop_deployment() {
    if [ "$USE_COMPOSE" = true ]; then
        print_color "$YELLOW" "Stopping Docker Compose services..."
        docker-compose down
    else
        print_color "$YELLOW" "Scaling down Kubernetes deployment..."
        kubectl scale deployment markdown-web-browser -n "$NAMESPACE" --replicas=0
    fi
    print_color "$GREEN" "✓ Services stopped"
}

# Clean up resources
cleanup() {
    if [ "$USE_COMPOSE" = true ]; then
        print_color "$YELLOW" "Cleaning Docker Compose resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
    else
        print_color "$YELLOW" "Cleaning Kubernetes resources..."
        kubectl delete all -n "$NAMESPACE" -l app=mdwb
    fi
    print_color "$GREEN" "✓ Resources cleaned"
}

# Main execution
main() {
    print_color "$BLUE" "═══════════════════════════════════════════"
    print_color "$BLUE" "  Markdown Web Browser Deployment"
    print_color "$BLUE" "═══════════════════════════════════════════"
    print_color "$YELLOW" "  Environment: $ENVIRONMENT"
    print_color "$YELLOW" "  Action: $ACTION"
    print_color "$YELLOW" "  Method: $([ "$USE_COMPOSE" = true ] && echo "Docker Compose" || echo "Kubernetes")"

    if [ "$DRY_RUN" = true ]; then
        print_color "$YELLOW" "  Mode: DRY RUN"
    fi

    echo

    case $ACTION in
        deploy)
            build_image
            if [ "$USE_COMPOSE" = true ]; then
                deploy_compose
            else
                deploy_kubernetes
            fi
            ;;
        update)
            build_image
            if [ "$USE_COMPOSE" = true ]; then
                docker-compose pull
                docker-compose up -d
            else
                kubectl set image deployment/markdown-web-browser \
                    web=markdown-web-browser:${IMAGE_TAG} \
                    -n "$NAMESPACE"
                kubectl rollout status deployment/markdown-web-browser -n "$NAMESPACE"
            fi
            print_color "$GREEN" "✓ Update complete"
            ;;
        rollback)
            if [ "$USE_COMPOSE" = true ]; then
                print_color "$RED" "Rollback not supported with Docker Compose"
                exit 1
            else
                kubectl rollout undo deployment/markdown-web-browser -n "$NAMESPACE"
                kubectl rollout status deployment/markdown-web-browser -n "$NAMESPACE"
            fi
            print_color "$GREEN" "✓ Rollback complete"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        stop)
            stop_deployment
            ;;
        clean)
            cleanup
            ;;
        *)
            print_color "$RED" "Unknown action: $ACTION"
            usage
            ;;
    esac

    echo
    print_color "$GREEN" "═══════════════════════════════════════════"
    print_color "$GREEN" "  Deployment Complete!"
    print_color "$GREEN" "═══════════════════════════════════════════"
}

# Run main function
main