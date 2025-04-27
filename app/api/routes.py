# app/api/routes.py
import os
import io
import time
import tempfile
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import FileResponse, StreamingResponse

from app.api.schemas import (
    SpeechRequest,
    AudioResponseFormat,
    AudioModelsResponse,
    AudioModelInfo,
    VoiceResponseFormat,
)
from app.services.dia_service import DiaService, DEFAULT_SAMPLE_RATE
from app.services.audio_service import AudioService
from app.models.voice import Voice # Assume this works as used elsewhere
from app.utils.helpers import validate_speed # Keep other helpers if they exist
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_dia_service():
    """Dependency for DiaService"""
    # This ensures the singleton instance is used
    return DiaService()

def get_audio_service():
    """Dependency for AudioService"""
    return AudioService(get_dia_service())

# OpenAI-compatible TTS endpoint
@router.post("/audio/speech")
async def create_speech(
    request: SpeechRequest,
    dia_service: DiaService = Depends(get_dia_service),
    audio_service: AudioService = Depends(get_audio_service),
):
    """Generate speech audio from text (OpenAI-compatible endpoint)"""
    try:
        logger.info(f"Received speech request for model: {request.model}, voice: {request.voice}")

        # --- Audio Prompt Logic ---
        audio_prompt_bytes: Optional[bytes] = None
        explicit_prompt_provided = False

        # 1. Check for explicit base64 audio prompt in the request
        if request.audio_prompt:
            logger.info("Explicit audio_prompt provided in request.")
            audio_prompt_bytes = audio_service.process_base64_audio(request.audio_prompt)
            if audio_prompt_bytes is None:
                raise HTTPException(status_code=400, detail="Invalid base64 audio_prompt provided.")
            explicit_prompt_provided = True

        # 2. If no explicit prompt, check if a custom voice is requested
        standard_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if not explicit_prompt_provided and request.voice not in standard_voices:
            logger.info(f"Custom voice '{request.voice}' requested, looking for sample.")
            try:
                voice_meta = Voice.from_file(request.voice)
                if voice_meta.samples:
                    # Use the first sample associated with the voice
                    sample_relative_path = voice_meta.samples[0]
                    sample_full_path = settings.STATIC_DIR / sample_relative_path
                    if sample_full_path.exists():
                        logger.info(f"Using sample file: {sample_full_path}")
                        with open(sample_full_path, "rb") as f_sample:
                            audio_prompt_bytes = f_sample.read()
                    else:
                        logger.warning(f"Sample file not found for custom voice '{request.voice}': {sample_full_path}")
                        # Proceed without prompt, or raise error depending on desired behavior
                        # raise HTTPException(status_code=404, detail=f"Sample file for voice '{request.voice}' not found.")
                else:
                    logger.warning(f"Custom voice '{request.voice}' has no associated samples.")
                    # Proceed without prompt
            except ValueError:
                # Voice ID doesn't match a custom voice file either
                logger.warning(f"Voice ID '{request.voice}' not found as standard or custom voice.")
                # Allow proceeding without prompt for potentially default behavior
                # Or raise an error:
                # raise HTTPException(status_code=400, detail=f"Invalid voice: '{request.voice}'. Choose a standard voice or provide a valid custom voice ID.")
            except Exception as e:
                logger.error(f"Error loading custom voice '{request.voice}': {e}")
                raise HTTPException(status_code=500, detail=f"Error processing custom voice '{request.voice}'.")

        # Determine max_tokens
        max_tokens = request.max_tokens if request.max_tokens else settings.DIA_DEFAULT_MAX_TOKENS
        logger.info(f"Using max_tokens: {max_tokens}")

        # --- Generate Speech ---
        start_time = time.time()
        generated_audio_np = dia_service.generate_speech(
            text=request.input,
            audio_prompt_bytes=audio_prompt_bytes,
            seed=request.seed,
            max_tokens=max_tokens
        )
        gen_time = time.time() - start_time
        logger.info(f"Speech generation took {gen_time:.2f} seconds.")

        if generated_audio_np is None:
             raise HTTPException(status_code=500, detail="Speech generation failed, produced no output.")

        # --- Post-processing (Speed Adjustment - Placeholder) ---
        speed = validate_speed(request.speed)
        if speed != 1.0:
             # Dia doesn't support speed adjustment natively in generate.
             # If implemented here, it would be post-processing.
             # generated_audio_np = audio_service.adjust_audio_speed(generated_audio_np, speed)
             logger.warning(f"Speed parameter ({speed}) is ignored as Dia generation doesn't support it directly.")

        # --- Format Conversion ---
        output_format = request.response_format.value
        logger.info(f"Converting audio to format: {output_format}")

        audio_bytes = audio_service.convert_audio_format(
            generated_audio_np,
            output_format,
            dia_service.get_sample_rate()
        )

        content_types = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "ogg": "audio/ogg", # Add ogg if supported by soundfile/ffmpeg
        }
        content_type = content_types.get(output_format, "application/octet-stream")

        # --- Return Response ---
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{output_format}"
            }
        )

    except HTTPException as he:
        # Re-raise HTTPExceptions directly
        raise he
    except ValueError as ve:
        logger.error(f"Value error during speech generation: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnfe:
         logger.error(f"File not found error: {str(fnfe)}")
         raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        logger.exception(f"Unexpected error generating speech: {str(e)}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

# Audio models endpoint (OpenAI compatibility)
@router.get("/audio/models", response_model=AudioModelsResponse)
async def list_models():
    """List available audio models"""
    # In the future, this could dynamically list models if more are supported
    models = [
        AudioModelInfo(
            id="dia-1.6b",
            name="Dia 1.6B",
            description="A 1.6B parameter text-to-speech model created by Nari Labs.",
            created=int(time.time()), # Or a fixed timestamp of model release
            owned_by="nari-labs",
        )
    ]
    return AudioModelsResponse(data=models, object="list")

# Audio response formats endpoint (Enhanced)
@router.get("/audio/speech/response-formats")
async def list_response_formats():
    """List supported audio response formats"""
    # Based on common soundfile/ffmpeg capabilities
    formats = [
        VoiceResponseFormat(name="mp3", content_type="audio/mpeg"),
        VoiceResponseFormat(name="wav", content_type="audio/wav"),
        VoiceResponseFormat(name="opus", content_type="audio/opus"),
        VoiceResponseFormat(name="flac", content_type="audio/flac"),
        VoiceResponseFormat(name="aac", content_type="audio/aac"), # Requires ffmpeg usually
        VoiceResponseFormat(name="ogg", content_type="audio/ogg"), # Vorbis commonly
    ]
    # Filter based on actual availability if needed (e.g., check ffmpeg)
    return {"response_formats": list(formats)}