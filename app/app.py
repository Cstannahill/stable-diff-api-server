import torch, asyncio, io, os, logging, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from diffusers import DiffusionPipeline  # type: ignore[reportPrivateImportUsage]
from typing import Optional
import uuid
from queue import Queue
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRequest(BaseModel):
    prompt: str
    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


class QueueStatus(BaseModel):
    request_id: str
    status: str  # "queued", "processing", "completed", "failed"
    position: Optional[int] = None
    estimated_wait_seconds: Optional[float] = None
    result_available: bool = False


app = FastAPI()

# Request queue management (initialized in startup event)
request_queue = None
active_requests = {}  # Track request status
queue_lock = Lock()

# Define local cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
MODEL_NAME = "black-forest-labs/FLUX.1-Krea-dev"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Load model with local caching and optimizations
logger.info(f"Loading model {MODEL_NAME} from cache directory: {CACHE_DIR}")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to("cuda" if torch.cuda.is_available() else "cpu")

# Optimize model with torch.compile and memory format
if torch.cuda.is_available():
    logger.info("Applying torch.compile optimizations...")
    # Set memory format for better GPU utilization
    pipe.transformer.to(memory_format=torch.channels_last)
    # Apply torch.compile for significant speedup
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    # Warmup inference to eliminate first-request latency
    logger.info("Performing warmup inference...")
    with torch.no_grad():
        _ = pipe(  # type: ignore[reportCallIssue]
            "warmup", guidance_scale=3.5, num_inference_steps=1, width=512, height=512
        )
    logger.info("Warmup completed")

logger.info("Model loaded and optimized successfully")


# Start background queue processor
async def process_queue():
    """Background task to process the request queue"""
    while True:
        try:
            # Get next request from queue
            if request_queue is None:
                await asyncio.sleep(1)
                continue
            request_data = await request_queue.get()
            request_id = request_data["request_id"]

            # Update status to processing
            with queue_lock:
                if request_id in active_requests:
                    active_requests[request_id]["status"] = "processing"

            logger.info(f"Processing request {request_id}")

            try:
                # Process the image generation
                result, generation_time = await asyncio.to_thread(
                    run_pipe,
                    request_data["prompt"],
                    request_data["guidance_scale"],
                    request_data["num_inference_steps"],
                    request_data["width"],
                    request_data["height"],
                    request_data["seed"],
                    True,
                )

                # Convert to PNG
                img = result.images[0]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)

                # Store result
                with queue_lock:
                    if request_id in active_requests:
                        active_requests[request_id].update(
                            {
                                "status": "completed",
                                "result": buf.getvalue(),
                                "generation_time": generation_time,
                                "completed_at": time.time(),
                            }
                        )

                logger.info(f"Request {request_id} completed in {generation_time:.2f}s")

            except Exception as e:
                logger.error(f"Error processing request {request_id}: {str(e)}")
                with queue_lock:
                    if request_id in active_requests:
                        active_requests[request_id].update(
                            {
                                "status": "failed",
                                "error": str(e),
                                "completed_at": time.time(),
                            }
                        )

            finally:
                if request_queue is not None:
                    request_queue.task_done()

        except Exception as e:
            logger.error(f"Queue processor error: {str(e)}")
            await asyncio.sleep(1)


# Cleanup old completed requests
async def cleanup_old_requests():
    """Clean up old completed/failed requests to prevent memory buildup"""
    while True:
        try:
            current_time = time.time()
            with queue_lock:
                # Remove requests older than 10 minutes
                old_requests = [
                    req_id
                    for req_id, req_data in active_requests.items()
                    if req_data.get("completed_at", 0) < current_time - 600
                    and req_data["status"] in ["completed", "failed"]
                ]
                for req_id in old_requests:
                    del active_requests[req_id]

                if old_requests:
                    logger.info(f"Cleaned up {len(old_requests)} old requests")

        except Exception as e:
            logger.error(f"Cleanup task error: {str(e)}")

        await asyncio.sleep(300)  # Run every 5 minutes


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    global request_queue
    request_queue = asyncio.Queue(maxsize=10)  # Initialize async queue
    asyncio.create_task(process_queue())
    asyncio.create_task(cleanup_old_requests())
    logger.info("Background queue processor started")


def run_pipe(
    prompt: str,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    return_dict: bool = True,
):
    logger.info(
        f"Starting image generation for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}' "
        f"[steps: {num_inference_steps}, guidance: {guidance_scale}, size: {width}x{height}]"
    )
    start_time = time.time()

    try:
        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            generator.manual_seed(seed)

        result = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            return_dict=return_dict,
        )  # type: ignore

        generation_time = time.time() - start_time
        logger.info(f"Image generation completed in {generation_time:.2f} seconds")
        return result, generation_time
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/queue/status")
async def queue_status():
    """Get current queue status"""
    queue_size = request_queue.qsize() if request_queue else 0
    with queue_lock:
        active_count = len(
            [r for r in active_requests.values() if r["status"] == "processing"]
        )

    return {
        "queue_size": queue_size,
        "active_requests": active_count,
        "total_slots": request_queue.maxsize if request_queue else 10,
    }


@app.get("/queue/status/{request_id}")
async def get_request_status(request_id: str):
    """Get status of specific request"""
    with queue_lock:
        if request_id not in active_requests:
            raise HTTPException(status_code=404, detail="Request not found")

        request_info = active_requests[request_id]

        # Calculate position in queue if still queued
        position = None
        estimated_wait = None
        if request_info["status"] == "queued":
            # Estimate position and wait time
            position = request_queue.qsize() if request_queue else 0
            estimated_wait = position * 15.0  # Estimate 15 seconds per request

        return QueueStatus(
            request_id=request_id,
            status=request_info["status"],
            position=position,
            estimated_wait_seconds=estimated_wait,
            result_available=request_info["status"] == "completed",
        )


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """Get the result of a completed image generation request"""
    with queue_lock:
        if request_id not in active_requests:
            raise HTTPException(status_code=404, detail="Request not found")

        request_info = active_requests[request_id]

        if request_info["status"] == "completed":
            # Return the generated image
            image_data = request_info["result"]
            buf = io.BytesIO(image_data)
            return StreamingResponse(buf, media_type="image/png")
        elif request_info["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Image generation failed: {request_info.get('error', 'Unknown error')}",
            )
        else:
            raise HTTPException(
                status_code=202,
                detail=f"Request is {request_info['status']}. Check status endpoint for updates.",
            )


@app.post("/generate/queue")
async def generate_queue(req: ImageRequest):
    """Queue a new image generation request"""
    try:
        # Check if queue is full
        if request_queue is None or request_queue.full():
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Queue is full. Please try again later.",
            )

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        logger.info(
            f"Queuing generation request {request_id} for prompt: '{req.prompt[:100]}{'...' if len(req.prompt) > 100 else ''}'"
        )

        # Add to active requests tracking
        with queue_lock:
            active_requests[request_id] = {
                "status": "queued",
                "created_at": time.time(),
                "prompt": req.prompt[:100] + ("..." if len(req.prompt) > 100 else ""),
            }

        # Add request to queue
        request_data = {
            "request_id": request_id,
            "prompt": req.prompt,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.num_inference_steps,
            "width": req.width,
            "height": req.height,
            "seed": req.seed,
        }

        if request_queue is not None:
            await request_queue.put(request_data)

        # Return request ID and status
        return {
            "request_id": request_id,
            "status": "queued",
            "message": "Request queued for processing",
            "queue_position": request_queue.qsize() if request_queue else 0,
            "estimated_wait_seconds": (request_queue.qsize() if request_queue else 0)
            * 15,
            "status_url": f"/queue/status/{request_id}",
            "result_url": f"/result/{request_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to queue request: {str(e)}"
        )


@app.post("/generate")
async def generate(req: ImageRequest):
    """Synchronous generation endpoint (original behavior)"""
    try:
        logger.info(
            f"Received synchronous generation request for prompt: '{req.prompt[:100]}{'...' if len(req.prompt) > 100 else ''}'"
        )

        result, generation_time = await asyncio.to_thread(
            run_pipe,
            req.prompt,
            req.guidance_scale,
            req.num_inference_steps,
            req.width,
            req.height,
            req.seed,
            True,
        )

        logger.info("Converting image to PNG format")
        img = result.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        logger.info(
            f"Synchronous image generation completed in {generation_time:.2f} seconds"
        )
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Error in synchronous generate endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )
