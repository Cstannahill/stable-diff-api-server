# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based REST API service for generating images using the FLUX.1-Krea-dev diffusion model from Black Forest Labs. The service provides a simple HTTP interface for text-to-image generation with GPU acceleration support.

## Architecture

- **FastAPI Server** (`app/app.py`): Main application entry point with two endpoints:
  - `GET /ping`: Health check endpoint
  - `POST /generate`: Image generation endpoint accepting a prompt and returning PNG image
- **Model Management**: Uses Hugging Face Diffusers library with local model caching in `app/model_cache/`
- **Containerization**: Docker setup with GPU support via docker-compose
- **Async Processing**: Image generation runs in background threads to avoid blocking the API

## Development Commands

### Running the Service

#### Local Development
```bash
# Local development (requires Python 3.11+)
pip install -r requirements.txt
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

#### Docker Deployment (Optimized)
```bash
# Production (optimized for RTX 4070 Super)
docker-compose -f docker-compose.optimized.yml up --build

# Development with hot reload
docker-compose -f docker-compose.dev.yml up --build

# With monitoring (Prometheus)
docker-compose -f docker-compose.optimized.yml --profile monitoring up

# Run optimization script
./scripts/docker-optimize.sh
```

#### Original Docker (Basic)
```bash
# Basic Docker setup (not optimized)
docker-compose up --build
```

### Testing the API

#### Synchronous Generation (Original)
```bash
# Health check
curl http://localhost:8000/ping

# Generate image synchronously (blocks until complete)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains", "guidance_scale": 3.5, "num_inference_steps": 28}' \
  --output generated_image.png
```

#### Asynchronous Queue System (New)
```bash
# Queue a generation request
curl -X POST http://localhost:8000/generate/queue \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains", "width": 1024, "height": 1024, "seed": 42}'

# Check queue status
curl http://localhost:8000/queue/status

# Check specific request status (use request_id from queue response)
curl http://localhost:8000/queue/status/{request_id}

# Get completed result (use request_id from queue response)
curl http://localhost:8000/result/{request_id} --output result.png
```

## Key Technical Details

### Model Loading & Optimization
- Model is loaded once at startup and cached locally in `app/model_cache/`
- **Performance Optimizations Applied**:
  - Uses `bfloat16` precision for better stability and performance on modern GPUs
  - `torch.compile` with `max-autotune` mode for 50%+ speedup
  - `channels_last` memory format for optimized GPU utilization
  - Warmup inference to eliminate first-request latency
- Automatically detects GPU availability and adjusts optimizations accordingly

### API Endpoints
- `POST /generate` - Synchronous generation (original behavior)
- `POST /generate/queue` - Queue-based async generation
- `GET /queue/status` - Overall queue status
- `GET /queue/status/{request_id}` - Individual request status
- `GET /result/{request_id}` - Retrieve completed results

### Request Parameters
- `prompt`: Text description for image generation
- `guidance_scale`: Controls prompt adherence (default: 3.5, optimal for FLUX)
- `num_inference_steps`: Quality vs speed tradeoff (default: 28)
- `width/height`: Image dimensions (default: 1024x1024)
- `seed`: For reproducible generation (optional)

### Performance Characteristics
- **Hardware Requirements**: RTX 4070 Super (12GB) or equivalent
- **Optimized Performance**: 2-8 seconds per image (60-75% faster than baseline)
- **Queue System**: Handles up to 10 concurrent requests with status tracking
- **Memory Management**: Automatic cleanup of completed requests after 10 minutes

### Docker Optimizations
- **Multi-stage builds** for smaller production images
- **CUDA 12.6 base** with cuDNN 9 for optimal GPU performance
- **Security hardening** with non-root user and minimal privileges
- **Memory optimization** with shared memory and proper limits
- **Development mode** with hot reload and debugging support
- **Health checks** and monitoring integration ready

### Error Handling
- Comprehensive logging throughout the generation pipeline
- HTTP 500 errors with detailed messages for generation failures
- Async processing prevents API timeouts during long generation times

## Dependencies

Core dependencies managed in `requirements.txt`:
- FastAPI + Uvicorn for web server
- Diffusers + Transformers for model inference
- PyTorch for deep learning backend
- Pillow for image processing
- Accelerate for optimized model loading