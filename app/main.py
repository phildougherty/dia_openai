# app/main.py
import os
import sys
import logging
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import settings
from app.api.routes import router as main_router
from app.api.voice_routes import router as voice_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Dia TTS API",
    description="OpenAI-compatible API for Dia text-to-speech",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Ensure required directories exist
settings.setup_directories()

# Add CORS middleware if enabled
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static files
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "An unexpected error occurred", "type": "server_error"}},
    )

# Redirect root to docs
@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Health check endpoint
@app.get("/health")
async def health_check():
    # Check if day service is initialized
    try:
        from app.services.dia_service import DiaService
        dia_service = DiaService()
        # Check if model is loaded
        sample_rate = dia_service.get_sample_rate()
        return {
            "status": "healthy",
            "model": "dia-1.6b",
            "sample_rate": sample_rate
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Version info endpoint
@app.get("/version")
async def version_info():
    return {
        "versions": {
            "api": "1.0.0",
            "model": "dia-1.6b"
        },
        "compatibility": {
            "openai": "v1"
        }
    }

# Include routers
# Include both under /api/v1 and /v1 paths for compatibility
app.include_router(main_router, prefix="/api/v1")
app.include_router(main_router, prefix="/v1")
app.include_router(voice_router, prefix="/api/v1")
app.include_router(voice_router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )