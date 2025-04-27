# app/utils/helpers.py
import random
import string
import re
from pathlib import Path
from typing import Optional, Tuple

# Note: prepare_audio_prompt was removed as its logic moved to AudioService

def generate_unique_id(length: int = 8) -> str:
    """Generate a random alphanumeric ID."""
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def sanitize_filename(filename: str) -> str:
    """Remove potentially problematic characters from a filename."""
    # Remove leading/trailing whitespace
    filename = filename.strip()
    # Replace spaces and problematic characters with underscores
    filename = re.sub(r'[\\/*?:"<>|\s]+', '_', filename)
    # Remove characters that are not alphanumeric, underscore, hyphen, or period
    filename = re.sub(r'[^\w.-]', '', filename)
    # Prevent filenames starting with '.'
    if filename.startswith('.'):
        filename = '_' + filename[1:]
    # Limit length (optional)
    max_len = 100
    if len(filename) > max_len:
        # Try to preserve extension
        base, ext = Path(filename).stem, Path(filename).suffix
        base = base[:max_len - len(ext)]
        filename = base + ext

    return filename if filename else "unnamed_file"

def validate_speed(speed: float) -> float:
    """Validate and clamp the speed value."""
    min_speed = 0.25
    max_speed = 4.0
    clamped_speed = max(min_speed, min(speed, max_speed))
    if clamped_speed != speed:
        print(f"Warning: Speed value {speed} clamped to {clamped_speed} (valid range: {min_speed}-{max_speed})")
    return clamped_speed

def format_voice_info(voice_obj, base_url: str) -> dict:
     """Helper to format voice info for API response (if Voice obj doesn't have it)"""
     # This is a placeholder assuming 'voice_obj' has attributes
     # like 'voice_id', 'name', 'description', 'samples', 'created_at'
     # Adapt based on the actual 'Voice' model structure
     return {
         "voice_id": getattr(voice_obj, 'voice_id', 'unknown'),
         "name": getattr(voice_obj, 'name', 'Unknown Name'),
         "description": getattr(voice_obj, 'description', None),
         "preview_url": f"{base_url}/api/v1/audio/voices/{getattr(voice_obj, 'voice_id', 'unknown')}/preview",
         "samples": [], # Populate this based on voice_obj.samples and _create_sample_info if needed
         "created_at": getattr(voice_obj, 'created_at', None),
     }

