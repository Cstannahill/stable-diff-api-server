# FLUX.1 API Server

A high-performance FastAPI server for AI image generation using the FLUX.1-Krea-dev model from Black Forest Labs. This server provides both synchronous and asynchronous endpoints for generating high-quality images from text prompts.

## 🚀 Features

- **FLUX.1-Krea-dev Model**: State-of-the-art text-to-image generation
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Local Model Caching**: Persistent model storage to avoid re-downloading
- **Dual API Modes**:
  - Synchronous generation (`/generate`) - Direct image response
  - Asynchronous queue system (`/generate/queue`) - For handling multiple requests
- **Performance Optimizations**:
  - `torch.compile` for faster inference
  - Memory-optimized tensor formats
  - Warmup inference to eliminate cold start latency
- **Comprehensive Logging**: Detailed request tracking and performance metrics
- **Queue Management**: Built-in request queuing with status tracking
- **Docker Support**: Multiple deployment configurations
- **Health Monitoring**: Built-in health checks and status endpoints

## 📋 Requirements

### System Requirements

- **GPU**: NVIDIA GPU with at least 12GB VRAM (RTX 4070 Super or better recommended)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: ~25GB for model cache
- **OS**: Linux (Ubuntu 20.04+ recommended), Windows 10/11, or macOS

### Software Requirements

- Python 3.10+
- NVIDIA CUDA Toolkit 11.8+ (for GPU acceleration)
- Docker & Docker Compose (for containerized deployment)

## 🛠️ Installation

### Option 1: Local Development Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Cstannahill/flux-api-server.git
   cd flux-api-server
   ```

2. **Create and activate virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   uvicorn app.app:app --host 127.0.0.1 --port 8000
   ```

### Option 2: Docker Deployment

#### Simple Docker Setup

```bash
docker-compose -f docker-compose.simple.yml up -d
```

#### Development with Hot Reload

```bash
docker-compose -f docker-compose.dev.yml up -d
```

#### Production Optimized

```bash
docker-compose -f docker-compose.optimized.yml up -d
```

## 🔧 Configuration

### Environment Variables

| Variable                  | Default                 | Description                                 |
| ------------------------- | ----------------------- | ------------------------------------------- |
| `MODEL_CACHE_DIR`         | `app/model_cache`       | Directory for storing cached models         |
| `LOG_LEVEL`               | `info`                  | Logging level (debug, info, warning, error) |
| `CUDA_VISIBLE_DEVICES`    | `0`                     | GPU device to use                           |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | CUDA memory allocation config               |

### Model Configuration

The server uses the `black-forest-labs/FLUX.1-Krea-dev` model by default. On first startup, the model (~24GB) will be downloaded and cached locally for subsequent runs.

## 📚 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### Health Check

```http
GET /ping
```

**Response:**

```json
{
  "status": "ok"
}
```

#### Synchronous Image Generation

```http
POST /generate
```

**Request Body:**

```json
{
  "prompt": "a beautiful sunset over mountains",
  "guidance_scale": 3.5,
  "num_inference_steps": 28,
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

**Parameters:**

- `prompt` (string, required): Text description of the desired image
- `guidance_scale` (float, optional): How closely to follow the prompt (default: 3.5)
- `num_inference_steps` (int, optional): Number of denoising steps (default: 28)
- `width` (int, optional): Image width in pixels (default: 1024)
- `height` (int, optional): Image height in pixels (default: 1024)
- `seed` (int, optional): Random seed for reproducible results

**Response:** PNG image file

#### Asynchronous Image Generation

```http
POST /generate/queue
```

**Request Body:** Same as synchronous endpoint

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Request queued for processing",
  "queue_position": 0,
  "estimated_wait_seconds": 0,
  "status_url": "/queue/status/550e8400-e29b-41d4-a716-446655440000",
  "result_url": "/result/550e8400-e29b-41d4-a716-446655440000"
}
```

#### Check Request Status

```http
GET /queue/status/{request_id}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "position": null,
  "estimated_wait_seconds": null,
  "result_available": true
}
```

#### Get Generated Image

```http
GET /result/{request_id}
```

**Response:** PNG image file (when status is "completed")

#### Queue Status

```http
GET /queue/status
```

**Response:**

```json
{
  "queue_size": 2,
  "active_requests": 1,
  "total_slots": 10
}
```

## 🚀 Usage Examples

### Python Client Example

```python
import requests
import io
from PIL import Image

# Synchronous generation
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "a majestic lion in a golden savanna at sunset",
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "width": 1024,
        "height": 1024
    }
)

if response.status_code == 200:
    image = Image.open(io.BytesIO(response.content))
    image.save("generated_image.png")
    print("Image saved successfully!")
```

### Asynchronous Generation Example

```python
import requests
import time

# Submit request to queue
response = requests.post(
    "http://localhost:8000/generate/queue",
    json={"prompt": "a cyberpunk cityscape at night"}
)

request_data = response.json()
request_id = request_data["request_id"]
status_url = f"http://localhost:8000/queue/status/{request_id}"
result_url = f"http://localhost:8000/result/{request_id}"

# Poll for completion
while True:
    status_response = requests.get(status_url)
    status = status_response.json()

    if status["status"] == "completed":
        # Download the result
        result_response = requests.get(result_url)
        with open("async_generated_image.png", "wb") as f:
            f.write(result_response.content)
        print("Async image generation completed!")
        break
    elif status["status"] == "failed":
        print("Generation failed")
        break
    else:
        print(f"Status: {status['status']}, Position: {status.get('position', 'N/A')}")
        time.sleep(2)
```

### cURL Examples

```bash
# Synchronous generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset over mountains"}' \
  --output sunset.png

# Queue a request
curl -X POST http://localhost:8000/generate/queue \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a futuristic robot"}' | jq .

# Check status
curl http://localhost:8000/queue/status/REQUEST_ID | jq .

# Download result
curl http://localhost:8000/result/REQUEST_ID --output result.png
```

## ⚡ Performance Optimization

### GPU Memory Management

The server is optimized for efficient GPU memory usage:

- Uses `torch.bfloat16` for reduced memory footprint
- Implements `torch.compile` for faster inference
- Optimized memory allocation with `PYTORCH_CUDA_ALLOC_CONF`

### Model Caching

- Models are cached locally in `app/model_cache/`
- First startup downloads ~24GB (one-time only)
- Subsequent startups load from cache (much faster)

### Generation Parameters for Speed vs Quality

- **Fast**: `num_inference_steps=20, guidance_scale=3.0`
- **Balanced**: `num_inference_steps=28, guidance_scale=3.5` (default)
- **High Quality**: `num_inference_steps=50, guidance_scale=4.0`

## 🐳 Docker Configurations

### Development Configuration (`docker-compose.dev.yml`)

- Hot reload enabled
- Source code mounted as volume
- Debug logging
- Development dependencies

### Simple Configuration (`docker-compose.simple.yml`)

- Basic production setup
- Essential optimizations
- Minimal resource allocation

### Optimized Configuration (`docker-compose.optimized.yml`)

- Production-ready with all optimizations
- Resource limits and health checks
- Optional Redis and monitoring services
- Security hardening

## 📊 Monitoring and Logging

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General operational information (default)
- **WARNING**: Warning messages
- **ERROR**: Error conditions

### Performance Metrics

The server logs include:

- Request processing times
- Queue statistics
- GPU memory usage
- Generation parameters

### Health Checks

- `/ping` endpoint for basic health monitoring
- Docker health checks included in production configurations

## 🔒 Security Considerations

- Non-root user execution in Docker
- Security options enabled
- No unnecessary privileges
- Input validation on all endpoints
- Resource limits to prevent abuse

## 🐛 Troubleshooting

### Common Issues

#### "CUDA out of memory"

- Reduce image dimensions (width/height)
- Lower `num_inference_steps`
- Restart the server to clear GPU memory

#### "Model not found" errors

- Ensure stable internet connection for initial download
- Check `MODEL_CACHE_DIR` permissions
- Verify sufficient disk space (~25GB)

#### Slow generation times

- Ensure GPU is properly configured
- Check GPU utilization with `nvidia-smi`
- Verify CUDA drivers are installed

#### Server won't start

- Check Python version (3.10+ required)
- Verify all dependencies are installed
- Check for port conflicts (default: 8000)

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=debug
uvicorn app.app:app --host 127.0.0.1 --port 8000 --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for the FLUX.1 model
- [Hugging Face](https://huggingface.co/) for the diffusers library
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Cstannahill/flux-api-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Cstannahill/flux-api-server/discussions)

---

**Note**: This project requires significant computational resources. A GPU with at least 12GB VRAM is recommended for optimal performance.
