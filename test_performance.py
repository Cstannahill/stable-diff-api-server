#!/usr/bin/env python3
"""
Performance test script for FLUX API optimizations
Tests both synchronous and asynchronous endpoints
"""

import requests
import time
import json
from typing import Dict, Any
import asyncio
import aiohttp

BASE_URL = "http://localhost:8000"

def test_sync_generation(prompt: str = "a beautiful sunset over mountains") -> Dict[str, Any]:
    """Test synchronous image generation"""
    print(f"Testing sync generation with prompt: '{prompt}'")
    
    payload = {
        "prompt": prompt,
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "width": 1024,
        "height": 1024,
        "seed": 42
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=120  # 2 minute timeout
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if response.status_code == 200:
            # Save image
            with open("test_sync_output.png", "wb") as f:
                f.write(response.content)
            
            return {
                "success": True,
                "generation_time": generation_time,
                "image_size": len(response.content),
                "message": f"Sync generation completed in {generation_time:.2f}s"
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "generation_time": generation_time
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "generation_time": time.time() - start_time
        }

async def test_async_generation(prompt: str = "a futuristic city at night") -> Dict[str, Any]:
    """Test asynchronous queue-based generation"""
    print(f"Testing async generation with prompt: '{prompt}'")
    
    payload = {
        "prompt": prompt,
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "width": 1024,
        "height": 1024,
        "seed": 123
    }
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Queue the request
            async with session.post(f"{BASE_URL}/generate/queue", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {"success": False, "error": f"Queue failed: {error_text}"}
                
                queue_result = await response.json()
                request_id = queue_result["request_id"]
                print(f"Request queued with ID: {request_id}")
            
            # Poll for completion
            max_wait = 180  # 3 minutes max
            poll_interval = 2  # Check every 2 seconds
            
            for _ in range(max_wait // poll_interval):
                async with session.get(f"{BASE_URL}/queue/status/{request_id}") as status_response:
                    if status_response.status == 200:
                        status_data = await status_response.json()
                        print(f"Status: {status_data['status']}")
                        
                        if status_data["status"] == "completed":
                            # Get the result
                            async with session.get(f"{BASE_URL}/result/{request_id}") as result_response:
                                if result_response.status == 200:
                                    image_data = await result_response.read()
                                    with open("test_async_output.png", "wb") as f:
                                        f.write(image_data)
                                    
                                    end_time = time.time()
                                    total_time = end_time - start_time
                                    
                                    return {
                                        "success": True,
                                        "generation_time": total_time,
                                        "image_size": len(image_data),
                                        "request_id": request_id,
                                        "message": f"Async generation completed in {total_time:.2f}s"
                                    }
                        
                        elif status_data["status"] == "failed":
                            return {
                                "success": False,
                                "error": "Generation failed",
                                "request_id": request_id
                            }
                
                await asyncio.sleep(poll_interval)
            
            return {
                "success": False,
                "error": "Timeout waiting for completion",
                "request_id": request_id
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "generation_time": time.time() - start_time
        }

def test_server_health():
    """Test server health and queue status"""
    try:
        # Ping test
        response = requests.get(f"{BASE_URL}/ping", timeout=10)
        if response.status_code != 200:
            return {"ping": False, "error": f"Ping failed: {response.status_code}"}
        
        # Queue status test
        queue_response = requests.get(f"{BASE_URL}/queue/status", timeout=10)
        if queue_response.status_code != 200:
            return {"ping": True, "queue_status": False, "error": "Queue status failed"}
        
        queue_data = queue_response.json()
        return {
            "ping": True,
            "queue_status": True,
            "queue_info": queue_data
        }
        
    except Exception as e:
        return {"ping": False, "error": str(e)}

async def main():
    print("=== FLUX API Performance Test ===\n")
    
    # Test server health
    print("1. Testing server health...")
    health_result = test_server_health()
    print(f"Health check: {json.dumps(health_result, indent=2)}\n")
    
    if not health_result.get("ping"):
        print("❌ Server is not responding. Make sure the API is running.")
        return
    
    # Test synchronous generation
    print("2. Testing synchronous generation...")
    sync_result = test_sync_generation()
    print(f"Sync result: {json.dumps(sync_result, indent=2)}\n")
    
    # Test asynchronous generation
    print("3. Testing asynchronous generation...")
    async_result = await test_async_generation()
    print(f"Async result: {json.dumps(async_result, indent=2)}\n")
    
    # Performance comparison
    print("=== Performance Summary ===")
    if sync_result.get("success") and async_result.get("success"):
        sync_time = sync_result["generation_time"]
        async_time = async_result["generation_time"]
        
        print(f"✅ Synchronous generation: {sync_time:.2f}s")
        print(f"✅ Asynchronous generation: {async_time:.2f}s")
        print(f"📊 Queue overhead: {async_time - sync_time:.2f}s")
        
        if sync_time < 15:  # Baseline expectation
            print(f"🚀 Performance optimizations working! (Target: <15s, Actual: {sync_time:.2f}s)")
        else:
            print(f"⚠️  Performance may need tuning (Expected: <15s, Actual: {sync_time:.2f}s)")
    else:
        if not sync_result.get("success"):
            print(f"❌ Synchronous generation failed: {sync_result.get('error')}")
        if not async_result.get("success"):
            print(f"❌ Asynchronous generation failed: {async_result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())