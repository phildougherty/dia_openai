# app/api/schemas.py
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, FilePath
from pathlib import Path

# Enums
class AudioResponseFormat(str, Enum):
    mp3 = "mp3"
    wav = "wav"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    ogg = "ogg" # Added ogg

class VoiceResponseFormat(BaseModel):
    """Voice response format"""
    name: str
    content_type: str

# Request models
class SpeechRequest(BaseModel):
    """OpenAI-compatible speech request format"""
    model: str = Field("dia-1.6b", description="The TTS model to use (currently only 'dia-1.6b' supported)")
    input: str = Field(..., description="The text to generate audio for. Use [S1] and [S2] for speaker turns.")
    voice: str = Field("alloy", description="The voice to use. Can be a standard voice (alloy, echo, fable, onyx, nova, shimmer) or a custom voice ID.")
    response_format: Optional[AudioResponseFormat] = Field(AudioResponseFormat.mp3, description="The format of the audio response")
    speed: float = Field(1.0, description="The speaking speed (Note: Currently ignored by Dia backend)", ge=0.25, le=4.0)

    # Dia-specific parameters (Optional)
    audio_prompt: Optional[str] = Field(None, description="Base64-encoded audio WAV data to use as a prompt for voice cloning. If provided, overrides the voice sample associated with the `voice` ID.")
    seed: Optional[int] = Field(None, description="Random seed for potentially increasing reproducibility.")
    max_tokens: Optional[int] = Field(None, description="Maximum number of audio tokens to generate. Overrides server default. More tokens = longer audio.")

    class Config:
        populate_by_name = True # Allows using 'input' instead of 'text' etc. if needed elsewhere
        extra = 'ignore' # Ignore extra fields passed in the request


# Response models
class VoiceSampleInfo(BaseModel):
    """Information about a voice sample file"""
    file_name: str
    content_type: str # e.g., "audio/wav"
    size_bytes: Optional[int] = None
    url: str # URL to download/access the sample

class VoiceInfo(BaseModel):
    """Voice information (Combined standard and custom)"""
    voice_id: str
    name: str
    description: Optional[str] = None
    preview_url: Optional[str] = None
    # For custom voices primarily
    samples: Optional[List[VoiceSampleInfo]] = Field(default_factory=list)
    created_at: Optional[str] = None # ISO format string

class VoiceListResponse(BaseModel):
    """Voice list response"""
    voices: List[VoiceInfo]

class AudioModelInfo(BaseModel):
    """Audio model information"""
    id: str
    name: str
    description: str
    created: int # Unix timestamp
    owned_by: str

class AudioModelsResponse(BaseModel):
    """Audio models response"""
    data: List[AudioModelInfo]
    object: str = "list"

class ErrorDetail(BaseModel):
    message: str
    type: Optional[str] = None
    param: Optional[str] = None
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: ErrorDetail

class StatusResponse(BaseModel):
    """Simple status response"""
    status: str
    message: Optional[str] = None