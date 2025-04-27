# app/api/voice_routes.py
import os
import io
import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Depends, File, Form, UploadFile, Query, Response as FastAPIResponse
from fastapi.responses import FileResponse

from app.api.schemas import VoiceInfo, VoiceListResponse, VoiceSampleInfo, StatusResponse
from app.services.dia_service import DiaService, DEFAULT_SAMPLE_RATE
from app.services.audio_service import AudioService
from app.models.voice import Voice # Assume exists and works
from app.utils.helpers import generate_unique_id, sanitize_filename, format_voice_info # Assume format_voice_info exists
from app.core.config import settings

router = APIRouter(prefix="/audio/voices", tags=["Voices"])
logger = logging.getLogger(__name__)

def get_dia_service():
    """Dependency for DiaService"""
    return DiaService()

def get_audio_service():
    """Dependency for AudioService"""
    return AudioService(get_dia_service())


def _create_sample_info(sample_relative_path: str, base_url: str) -> Optional[VoiceSampleInfo]:
    """Helper to create VoiceSampleInfo from a relative path"""
    try:
        full_path = (settings.STATIC_DIR / sample_relative_path).resolve()
        if not full_path.is_file():
            logger.warning(f"Sample file not found at expected path: {full_path}")
            return None

        # Basic content type detection based on extension
        content_type = "audio/wav" # Default assumption for samples
        if full_path.suffix.lower() == ".mp3":
            content_type = "audio/mpeg"
        elif full_path.suffix.lower() == ".ogg":
            content_type = "audio/ogg"
        elif full_path.suffix.lower() == ".flac":
            content_type = "audio/flac"

        return VoiceSampleInfo(
            file_name=full_path.name,
            content_type=content_type,
            size_bytes=full_path.stat().st_size,
            url=f"{base_url}/static/{sample_relative_path}"
        )
    except Exception as e:
        logger.error(f"Error processing sample path '{sample_relative_path}': {e}")
        return None

# List voices (OpenAI compatibility & Custom Voices)
@router.get("", response_model=VoiceListResponse)
async def list_voices(request: Request):
    """List all available voices (standard and custom)"""
    try:
        base_url = str(request.base_url).rstrip('/')
        all_voices = []

        # Standard voices
        standard_voice_data = {
            "alloy": "A balanced and natural tone",
            "echo": "A resonant and deeper voice",
            "fable": "A bright, higher-pitched voice",
            "onyx": "A deep and authoritative voice",
            "nova": "A warm and smooth voice",
            "shimmer": "A light and airy voice",
        }
        for voice_id, description in standard_voice_data.items():
            all_voices.append(VoiceInfo(
                voice_id=voice_id,
                name=voice_id.capitalize(),
                description=f"Standard voice: {description}",
                preview_url=f"{base_url}/api/v1/audio/voices/{voice_id}/preview" # Ensure prefix is correct
            ))

        # Custom voices from files
        custom_voice_objs = Voice.list_voices()
        for voice in custom_voice_objs:
            # Reconstruct VoiceInfo for custom voices properly
            sample_infos = []
            for sample_path_str in voice.samples:
                sample_info = _create_sample_info(sample_path_str, base_url)
                if sample_info:
                    sample_infos.append(sample_info)

            all_voices.append(VoiceInfo(
                voice_id=voice.voice_id,
                name=voice.name,
                description=voice.description,
                preview_url=f"{base_url}/api/v1/audio/voices/{voice.voice_id}/preview",
                samples=sample_infos,
                created_at=voice.created_at # Assuming Voice model has this
            ))

        return VoiceListResponse(voices=all_voices)
    except Exception as e:
        logger.exception("Error listing voices")
        raise HTTPException(status_code=500, detail=f"Error listing voices: {str(e)}")


# Get a specific voice
@router.get("/{voice_id}", response_model=VoiceInfo)
async def get_voice(voice_id: str, request: Request):
    """Get a specific voice by ID"""
    try:
        base_url = str(request.base_url).rstrip('/')
        standard_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

        if voice_id in standard_voices:
            # Return info for standard voice
            voice_name = voice_id.capitalize()
            description = { # Descriptions copied from list_voices
                "alloy": "A balanced and natural tone", "echo": "A resonant and deeper voice",
                "fable": "A bright, higher-pitched voice", "onyx": "A deep and authoritative voice",
                "nova": "A warm and smooth voice", "shimmer": "A light and airy voice",
            }.get(voice_id, "Standard voice")
            return VoiceInfo(
                voice_id=voice_id,
                name=voice_name,
                description=f"Standard voice: {description}",
                preview_url=f"{base_url}/api/v1/audio/voices/{voice_id}/preview"
            )
        else:
            # Try to load custom voice
            try:
                voice = Voice.from_file(voice_id)
                 # Reconstruct VoiceInfo including samples
                sample_infos = []
                for sample_path_str in voice.samples:
                    sample_info = _create_sample_info(sample_path_str, base_url)
                    if sample_info:
                        sample_infos.append(sample_info)

                return VoiceInfo(
                    voice_id=voice.voice_id,
                    name=voice.name,
                    description=voice.description,
                    preview_url=f"{base_url}/api/v1/audio/voices/{voice.voice_id}/preview",
                    samples=sample_infos,
                    created_at=voice.created_at # Assuming Voice model has this
                )
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error getting voice '{voice_id}'")
        raise HTTPException(status_code=500, detail=f"Error getting voice: {str(e)}")

# Get voice preview
@router.get("/{voice_id}/preview", response_class=FastAPIResponse)
async def get_voice_preview(
    voice_id: str,
    dia_service: DiaService = Depends(get_dia_service),
    audio_service: AudioService = Depends(get_audio_service),
):
    """Generate and return a preview audio for a specific voice"""
    try:
        logger.info(f"Generating preview for voice: {voice_id}")
        preview_text = f"[S1] This is a preview of the {voice_id} voice. [S2] It uses the Dia text-to-speech model."
        audio_prompt_bytes: Optional[bytes] = None
        seed: Optional[int] = 42 # Use a fixed seed for standard voice previews

        # Check if it's a custom voice and load its sample
        standard_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice_id not in standard_voices:
            seed = None # Use default random seed for custom previews unless specified
            try:
                voice = Voice.from_file(voice_id)
                if voice.samples:
                    sample_relative_path = voice.samples[0]
                    sample_full_path = settings.STATIC_DIR / sample_relative_path
                    if sample_full_path.exists():
                        logger.info(f"Using sample for preview prompt: {sample_full_path}")
                        with open(sample_full_path, "rb") as f_sample:
                            audio_prompt_bytes = f_sample.read()
                    else:
                        logger.warning(f"Sample file for preview not found: {sample_full_path}")
                        # Proceed without prompt for preview
                else:
                    logger.warning(f"Custom voice '{voice_id}' has no samples for preview prompt.")
                    # Proceed without prompt
            except ValueError:
                 raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found for preview generation")
            except Exception as e:
                logger.error(f"Error loading custom voice '{voice_id}' for preview: {e}")
                # Proceed without prompt maybe? Or raise error? Let's raise.
                raise HTTPException(status_code=500, detail=f"Error preparing preview for voice '{voice_id}'.")

        # Generate audio
        generated_audio_np = dia_service.generate_speech(
            text=preview_text,
            audio_prompt_bytes=audio_prompt_bytes,
            seed=seed
        )

        if generated_audio_np is None:
            raise HTTPException(status_code=500, detail="Preview generation failed.")

        # Convert to MP3
        audio_bytes = audio_service.convert_audio_format(
            generated_audio_np,
            "mp3",
             dia_service.get_sample_rate()
        )

        # Return the audio
        return FastAPIResponse(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename={sanitize_filename(voice_id)}_preview.mp3"
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error generating preview for voice '{voice_id}'")
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")

# Create a custom voice
@router.post("", response_model=VoiceInfo, status_code=201)
async def create_voice(
    request: Request,
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(..., description="Audio sample file (e.g., WAV) for the voice."),
    dia_service: DiaService = Depends(get_dia_service), # For validation
    audio_service: AudioService = Depends(get_audio_service), # For saving
):
    """Create a new custom voice from an audio sample"""
    try:
        logger.info(f"Received request to create voice: {name}")
        # Generate a unique ID for the new voice
        voice_id = f"custom_{generate_unique_id()}"
        logger.info(f"Generated voice ID: {voice_id}")

        # Read the uploaded file
        audio_content = await file.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        # Basic validation using DiaService (check if loadable)
        # This implicitly checks if the format is somewhat reasonable
        # We write to a temp file because Dia's loading often expects paths
        temp_suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=True) as temp_audio_file:
            temp_audio_file.write(audio_content)
            temp_audio_file.flush()
            logger.info(f"Validating uploaded audio file: {file.filename}")
            if not dia_service.validate_audio_prompt_file(temp_audio_file.name):
                 raise HTTPException(
                     status_code=400,
                     detail=f"Invalid or unsupported audio file format/content: {file.filename}"
                 )

        # Create the new voice object
        new_voice = Voice(
            voice_id=voice_id,
            name=name,
            description=description or f"Custom voice: {name}",
            samples=[], # Samples will be paths relative to STATIC_DIR
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

        # Save the voice metadata first
        new_voice.save() # Assumes this saves to voices/{voice_id}.json

        # Save the audio sample to static/audio
        # Sanitize filename before using parts of it
        safe_original_filename = sanitize_filename(Path(file.filename).stem if file.filename else "sample")
        sample_filename = f"{voice_id}_{safe_original_filename}{temp_suffix}" # e.g., custom_abc123_my_sample.wav
        saved_path = audio_service.save_audio_file(audio_content, sample_filename)

        if not saved_path:
            # Attempt cleanup if save failed after metadata save
            try:
                voice_meta_path = settings.VOICES_DIR / f"{voice_id}.json"
                if voice_meta_path.exists():
                    os.remove(voice_meta_path)
            except Exception as cleanup_err:
                 logger.error(f"Failed to cleanup voice metadata after sample save failure: {cleanup_err}")
            raise HTTPException(status_code=500, detail="Failed to save voice sample file.")

        # Add the relative path of the saved sample to the voice metadata
        relative_sample_path = str(saved_path.relative_to(settings.STATIC_DIR))
        new_voice.add_sample(relative_sample_path) # Assumes Voice model handles this
        new_voice.save() # Save metadata again with the sample path

        # Format the response
        base_url = str(request.base_url).rstrip('/')
        sample_infos = []
        sample_info = _create_sample_info(relative_sample_path, base_url)
        if sample_info:
            sample_infos.append(sample_info)

        return VoiceInfo(
            voice_id=new_voice.voice_id,
            name=new_voice.name,
            description=new_voice.description,
            preview_url=f"{base_url}/api/v1/audio/voices/{new_voice.voice_id}/preview",
            samples=sample_infos,
            created_at=new_voice.created_at
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error creating voice '{name}'")
        # Attempt cleanup if voice metadata might have been saved
        if 'voice_id' in locals():
            try:
                voice_meta_path = settings.VOICES_DIR / f"{voice_id}.json"
                if voice_meta_path.exists():
                    os.remove(voice_meta_path)
            except Exception as cleanup_err:
                 logger.error(f"Failed to cleanup voice metadata during error handling: {cleanup_err}")
        raise HTTPException(status_code=500, detail=f"Error creating voice: {str(e)}")

# Delete a custom voice
@router.delete("/{voice_id}", response_model=StatusResponse)
async def delete_voice(voice_id: str):
    """Delete a custom voice and its associated samples"""
    try:
        standard_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice_id in standard_voices:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete standard voice '{voice_id}'"
            )

        logger.info(f"Attempting to delete voice: {voice_id}")
        # Check if the voice metadata file exists
        voice_meta_path = settings.VOICES_DIR / f"{voice_id}.json"
        if not voice_meta_path.exists():
            raise HTTPException(status_code=404, detail=f"Custom voice '{voice_id}' not found")

        # Load the voice to get sample paths before deleting metadata
        try:
            voice = Voice.from_file(voice_id)
        except Exception as load_err:
             # Log error but proceed to delete metadata if file exists
             logger.error(f"Error loading voice metadata for {voice_id} during deletion, but will proceed: {load_err}")
             voice = None # Ensure samples aren't processed if load failed

        # Delete sample files associated with the voice
        if voice and voice.samples:
            for sample_relative_path in voice.samples:
                try:
                    sample_full_path = (settings.STATIC_DIR / sample_relative_path).resolve()
                    if sample_full_path.exists() and sample_full_path.is_file():
                        logger.info(f"Deleting sample file: {sample_full_path}")
                        os.remove(sample_full_path)
                    else:
                        logger.warning(f"Sample file not found or is not a file, skipping deletion: {sample_full_path}")
                except Exception as sample_del_err:
                    # Log error but continue trying to delete other files and metadata
                    logger.error(f"Error deleting sample file {sample_relative_path} for voice {voice_id}: {sample_del_err}")

        # Delete the voice metadata file
        logger.info(f"Deleting voice metadata file: {voice_meta_path}")
        os.remove(voice_meta_path)

        return StatusResponse(status="success", message=f"Voice '{voice_id}' deleted successfully")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error deleting voice '{voice_id}'")
        raise HTTPException(status_code=500, detail=f"Error deleting voice: {str(e)}")