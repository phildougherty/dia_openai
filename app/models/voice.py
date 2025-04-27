# app/models/voice.py
import os
import json
import logging # Import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__) # Setup logger

class Voice(BaseModel):
    """Model representing a voice"""
    voice_id: str
    name: str
    description: Optional[str] = None
    # Use timezone-aware datetime if consistency is critical, otherwise naive is often fine for simpler apps
    created_at: datetime = Field(default_factory=datetime.utcnow) # Use utcnow for consistency
    samples: List[str] = Field(default_factory=list) # Use default_factory for mutable defaults

    class Config:
        # Pydantic V2 uses model_config attribute
        json_encoders = {
            # Ensure datetime is always encoded to ISO format string with Z for UTC
            datetime: lambda v: v.isoformat(timespec='seconds') + 'Z' if v.tzinfo is None else v.isoformat(timespec='seconds')
        }
        validate_assignment = True # Good practice: validates fields on assignment

    @classmethod
    def from_file(cls, voice_id: str) -> 'Voice':
        """Load a voice from a JSON file"""
        voice_path = settings.VOICES_DIR / f"{voice_id}.json"

        if not voice_path.exists():
            logger.error(f"Voice file not found: {voice_path}")
            raise ValueError(f"Voice with ID {voice_id} not found")

        try:
            with open(voice_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Pydantic V2 will automatically parse the ISO string back to datetime
            return cls(**data)
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON from {voice_path}: {e}")
             raise ValueError(f"Invalid format in voice file for ID {voice_id}") from e
        except ValidationError as e:
             logger.error(f"Validation error loading voice {voice_id} from {voice_path}: {e}")
             raise ValueError(f"Data mismatch in voice file for ID {voice_id}") from e
        except Exception as e:
             logger.exception(f"Unexpected error loading voice {voice_id} from {voice_path}")
             raise # Re-raise unexpected errors

    def save(self) -> None:
        """Save voice model to a JSON file."""
        voice_path = settings.VOICES_DIR / f"{self.voice_id}.json"
        logger.info(f"Saving voice metadata to: {voice_path}")
        try:
            # Ensure the directory exists (though should be handled by config setup)
            settings.VOICES_DIR.mkdir(parents=True, exist_ok=True)

            # Use model_dump_json which respects Config.json_encoders
            voice_json = self.model_dump_json(indent=2)

            with open(voice_path, 'w', encoding='utf-8') as f:
                f.write(voice_json)
            logger.info(f"Successfully saved voice: {self.voice_id}")
        except IOError as e:
            logger.error(f"IOError saving voice file {voice_path}: {e}")
            # Depending on requirements, might want to raise an exception here
        except Exception as e:
            logger.exception(f"Unexpected error saving voice {self.voice_id} to {voice_path}")
            # Depending on requirements, might want to raise an exception here


    def add_sample(self, sample_path: str) -> None:
        """Add a sample audio file relative path to the voice samples list."""
        # Ensure path uses forward slashes for consistency, even on Windows
        normalized_path = Path(sample_path).as_posix()
        if normalized_path not in self.samples:
            logger.info(f"Adding sample '{normalized_path}' to voice '{self.voice_id}'")
            self.samples.append(normalized_path)
            self.save() # Save after modification
        else:
             logger.debug(f"Sample '{normalized_path}' already exists for voice '{self.voice_id}'")

    @staticmethod
    def list_voices() -> List['Voice']:
        """List all available voices by loading their JSON metadata files."""
        voices = []
        if not settings.VOICES_DIR.exists():
             logger.warning(f"Voices directory not found: {settings.VOICES_DIR}")
             return voices # Return empty list if dir doesn't exist

        logger.info(f"Listing voices from directory: {settings.VOICES_DIR}")
        for voice_file in settings.VOICES_DIR.glob("*.json"):
            try:
                voice_id = voice_file.stem
                logger.debug(f"Loading voice metadata for ID: {voice_id}")
                voice = Voice.from_file(voice_id)
                voices.append(voice)
            except ValueError as e: # Catch specific errors from from_file
                 logger.error(f"Skipping invalid voice file {voice_file.name}: {e}")
            except Exception as e: # Catch unexpected errors during loading
                # Use logger.exception to include traceback for unexpected errors
                logger.exception(f"Unexpected error loading voice file {voice_file.name}")

        logger.info(f"Found {len(voices)} valid voices.")
        # Optionally sort voices by name or ID
        # voices.sort(key=lambda v: v.name)
        return voices