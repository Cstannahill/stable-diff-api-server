#!/bin/bash
# Docker optimization script for FLUX API

set -e

echo "🚀 FLUX API Docker Optimization Script"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running!"
        exit 1
    fi
    
    print_success "Docker is running"
}

# Check NVIDIA Docker support
check_nvidia_docker() {
    print_status "Checking NVIDIA Docker support..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "NVIDIA drivers not found!"
        exit 1
    fi
    
    if ! docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker runtime not working!"
        print_warning "Please install nvidia-container-toolkit"
        exit 1
    fi
    
    print_success "NVIDIA Docker support is working"
}

# Optimize Docker daemon settings
optimize_docker_daemon() {
    print_status "Checking Docker daemon configuration..."
    
    DAEMON_CONFIG="/etc/docker/daemon.json"
    
    if [ -f "$DAEMON_CONFIG" ]; then
        print_status "Docker daemon.json exists, checking configuration..."
    else
        print_warning "Creating Docker daemon configuration..."
        sudo mkdir -p /etc/docker
        sudo tee "$DAEMON_CONFIG" > /dev/null <<EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
        print_success "Docker daemon configuration created"
        print_warning "Please restart Docker daemon: sudo systemctl restart docker"
    fi
}

# Build optimized image
build_optimized() {
    print_status "Building optimized Docker image..."
    
    # Enable BuildKit for better caching and performance
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    # Build with cache optimization
    docker build -f Dockerfile.optimized \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from flux-api:latest \
        -t flux-api:optimized .
    
    print_success "Optimized image built successfully"
}

# Clean up old images and containers
cleanup() {
    print_status "Cleaning up old Docker artifacts..."
    
    # Remove old containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful!)
    read -p "Remove unused Docker volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
        print_success "Volumes cleaned"
    fi
    
    print_success "Cleanup completed"
}

# Performance test
performance_test() {
    print_status "Running Docker performance test..."
    
    # Test GPU access
    docker run --rm --gpus all flux-api:optimized python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
    
    print_success "Performance test completed"
}

# Main execution
main() {
    echo
    print_status "Starting Docker optimization process..."
    
    check_docker
    check_nvidia_docker
    optimize_docker_daemon
    
    # Ask user what to do
    echo
    echo "What would you like to do?"
    echo "1) Build optimized image"
    echo "2) Clean up old Docker artifacts"
    echo "3) Run performance test"
    echo "4) All of the above"
    echo "5) Exit"
    
    read -p "Choose an option (1-5): " choice
    
    case $choice in
        1)
            build_optimized
            ;;
        2)
            cleanup
            ;;
        3)
            performance_test
            ;;
        4)
            build_optimized
            cleanup
            performance_test
            ;;
        5)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
    
    echo
    print_success "Docker optimization completed!"
    echo
    echo "🎯 Quick commands:"
    echo "  Production: docker-compose -f docker-compose.optimized.yml up"
    echo "  Development: docker-compose -f docker-compose.dev.yml up"
    echo "  With monitoring: docker-compose -f docker-compose.optimized.yml --profile monitoring up"
}

# Run main function
main "$@"