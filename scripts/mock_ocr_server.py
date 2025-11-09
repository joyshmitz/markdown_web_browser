#!/usr/bin/env python3
"""Mock OCR server for testing when the real olmOCR API is unavailable."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import hashlib
import uvicorn
import random
from datetime import datetime

app = FastAPI(title="Mock olmOCR Server")


class OCRInput(BaseModel):
    id: str
    image: str  # Base64 encoded image

class OCROptions(BaseModel):
    fp8: bool = False

class OCRRequest(BaseModel):
    model: str = "olmOCR-2-7B-1025-FP8"
    input: List[OCRInput]  # List of images with IDs
    options: OCROptions = OCROptions()


class OCRResponse(BaseModel):
    choices: List[Dict[str, Any]]


@app.post("/v1/completions")
@app.post("/v1/completions/v1/ocr")
async def process_ocr(request: OCRRequest):
    """Mock OCR endpoint that returns placeholder text."""

    responses = []

    for idx, input_item in enumerate(request.input):
        # Generate deterministic fake content based on image hash
        image_hash = hashlib.sha256(input_item.image.encode()).hexdigest()[:8]

        # Simulate realistic OCR output
        mock_text = f"""# Example Domain

This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.

[More information...](https://www.iana.org/domains/example)

---
*OCR processed tile {input_item.id} (hash: {image_hash})*"""

        responses.append({
            "index": idx,
            "message": {
                "role": "assistant",
                "content": mock_text
            },
            "finish_reason": "stop",
            "logprobs": None,
        })

    return OCRResponse(choices=responses)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "mock-olmOCR",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Mock olmOCR Server",
        "version": "1.0.0",
        "endpoints": {
            "/v1/completions": "POST - Process OCR requests",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation",
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mock OCR Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════╗
║       Mock olmOCR Server Starting      ║
╠════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port:<28} ║
║  Docs: http://{args.host}:{args.port}/docs{"":>21} ║
║  Health: http://{args.host}:{args.port}/health{"":>18} ║
╚════════════════════════════════════════╝
    """)

    uvicorn.run(
        "mock_ocr_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )