# app/services/audio_service.py
import os
import io
import base64
import tempfile
import subprocess
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, BinaryIO

from app.core.config import settings
# No direct DiaService dependency needed here anymore for conversion if done via soundfile
# from app.services.dia_service import DiaService

logger = logging.getLogger(__name__)

class AudioService:
    """Service for handling audio processing and conversion"""

    def __init__(self, dia_service):
        # Keep dia_service if needed for other things like validation or sample rate
        self.dia_service = dia_service

    def convert_audio_format(
        self,
        audio_np: np.ndarray, # Expect numpy array from DiaService
        output_format: str = "mp3",
        sample_rate: int = 44100
    ) -> bytes:
        """Convert numpy audio array to bytes in specified format using soundfile/ffmpeg."""
        logger.info(f"Attempting conversion to {output_format} at {sample_rate} Hz")
        output_format = output_format.lower()
        # Ensure audio is float32 for processing
        if audio_np.dtype != np.float32:
             # Assuming Dia outputs float32 in [-1, 1] range based on its app.py conversion
             # If it's int16, convert back: audio_np = audio_np.astype(np.float32) / 32767.0
             # Let's assume it IS float32 for now.
             if np.issubdtype(audio_np.dtype, np.integer):
                 max_val = np.iinfo(audio_np.dtype).max
                 audio_np = audio_np.astype(np.float32) / max_val
             else:
                 # Attempt conversion or raise error if unexpected type
                 try:
                    audio_np = audio_np.astype(np.float32)
                 except Exception as e:
                     logger.error(f"Cannot convert audio dtype {audio_np.dtype} to float32 for processing: {e}")
                     raise ValueError("Unsupported audio data type for conversion")

        audio_np = np.clip(audio_np, -1.0, 1.0) # Ensure range

        bytes_io = io.BytesIO()

        try:
            # Use soundfile for formats it supports directly (wav, flac, ogg/vorbis)
            if output_format in ["wav", "flac", "ogg"]:
                subtype = 'FLOAT' if output_format == 'wav' else None # Use default subtypes for others
                sf.write(bytes_io, audio_np, sample_rate, format=output_format.upper(), subtype=subtype)
                logger.info(f"Successfully converted audio to {output_format} using soundfile.")

            # Use ffmpeg (via subprocess) for formats like mp3, aac, opus
            elif output_format in ["mp3", "aac", "opus"]:
                logger.info(f"Using ffmpeg for {output_format} conversion.")
                # Write to temporary WAV file first as input for ffmpeg
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                    sf.write(temp_wav.name, audio_np, sample_rate, format='WAV', subtype='FLOAT')
                    temp_wav.flush() # Ensure data is written before ffmpeg reads

                    # Build ffmpeg command
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y", # Overwrite output file if it exists (shouldn't for BytesIO pipe)
                        "-i", temp_wav.name, # Input from temp WAV
                        "-vn", # No video
                        "-ar", str(sample_rate), # Audio sample rate
                        "-ac", "1", # Audio channels (assuming mono from Dia, adjust if stereo)
                        # Add specific codec options if needed (e.g., bitrate for mp3)
                        # "-b:a", "192k", # Example bitrate for MP3
                        "-f", output_format, # Output format muxer
                        "pipe:1" # Output to stdout
                    ]
                    logger.debug(f"Executing ffmpeg command: {' '.join(ffmpeg_cmd)}")

                    # Execute ffmpeg
                    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()

                    if process.returncode != 0:
                        ffmpeg_error = stderr.decode('utf-8', errors='ignore')
                        logger.error(f"ffmpeg conversion failed (code {process.returncode}):\n{ffmpeg_error}")
                        raise RuntimeError(f"ffmpeg failed to convert audio to {output_format}: {ffmpeg_error.splitlines()[-1]}")
                    else:
                         ffmpeg_log = stderr.decode('utf-8', errors='ignore')
                         logger.debug(f"ffmpeg output log:\n{ffmpeg_log}")

                    bytes_io.write(stdout)
                    logger.info(f"Successfully converted audio to {output_format} using ffmpeg.")

            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            # Get bytes from BytesIO
            bytes_io.seek(0)
            audio_bytes = bytes_io.read()
            if not audio_bytes:
                raise RuntimeError(f"Conversion to {output_format} produced empty output.")
            return audio_bytes

        except sf.LibsndfileError as sfe:
             logger.error(f"Soundfile error during conversion to {output_format}: {sfe}")
             raise RuntimeError(f"Audio library error during conversion: {sfe}") from sfe
        except FileNotFoundError as fnfe:
             if 'ffmpeg' in str(fnfe):
                 logger.error("ffmpeg command not found. Ensure ffmpeg is installed and in PATH.")
                 raise RuntimeError("ffmpeg is required for this audio format but was not found.") from fnfe
             else:
                 logger.error(f"File not found error during conversion: {fnfe}")
                 raise
        except Exception as e:
            logger.exception(f"Error converting audio to {output_format}: {e}")
            raise RuntimeError(f"Failed to convert audio to {output_format}") from e
        finally:
             bytes_io.close()


    def process_base64_audio(self, base64_string: str) -> Optional[bytes]:
        """Process base64 encoded audio string to bytes"""
        if not base64_string:
            logger.warning("Received empty base64 audio string.")
            return None
        try:
            # Remove data URI prefix if present (e.g., "data:audio/wav;base64,")
            if ',' in base64_string:
                _, base64_data = base64_string.split(',', 1)
            else:
                base64_data = base64_string

            # Decode the base64 string
            audio_bytes = base64.b64decode(base64_data)
            logger.info(f"Successfully decoded base64 audio string ({len(audio_bytes)} bytes).")
            return audio_bytes
        except base64.binascii.Error as b64e:
             logger.error(f"Invalid base64 encoding: {str(b64e)}")
             return None
        except Exception as e:
            logger.exception(f"Error processing base64 audio string")
            return None


    def save_audio_file(self, audio_content: bytes, filename: str) -> Optional[Path]:
        """Save audio bytes to a file in the static audio directory"""
        try:
            audio_dir = settings.STATIC_DIR / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

            # Sanitize filename before saving
            safe_filename = sanitize_filename(filename)
            file_path = audio_dir / safe_filename

            logger.info(f"Saving audio content to: {file_path}")
            with open(file_path, "wb") as f:
                f.write(audio_content)

            logger.info(f"Successfully saved audio file: {file_path}")
            return file_path
        except IOError as ioe:
             logger.error(f"I/O error saving audio file {filename}: {ioe}")
             return None
        except Exception as e:
            logger.exception(f"Error saving audio file {filename}")
            return None

    def adjust_audio_speed(
        self,
        audio_np: np.ndarray,
        speed: float = 1.0
    ) -> np.ndarray:
        """Adjust the speed of the audio using numpy interpolation (simple method)"""
        if speed == 1.0:
            return audio_np
        if speed <= 0:
             logger.warning("Speed must be positive, ignoring adjustment.")
             return audio_np

        logger.info(f"Adjusting audio speed by factor {speed:.2f}x")
        try:
           original_len = len(audio_np)
           target_len = int(original_len / speed)

           if target_len < 1:
               logger.warning("Resulting audio length too short after speed adjustment, returning original.")
               return audio_np

           # Create original and new time axes
           x_original = np.arange(original_len)
           # Use linspace for potentially more accurate interpolation points
           x_resampled = np.linspace(0, original_len - 1, target_len)

           # Interpolate
           resampled_audio_np = np.interp(x_resampled, x_original, audio_np)

           logger.info(f"Audio resampled from {original_len} to {target_len} samples for speed {speed:.2f}x")
           return resampled_audio_np.astype(audio_np.dtype)

        except Exception as e:
            logger.exception(f"Error adjusting audio speed")
            # Return original audio if adjustment fails
            return audio_np