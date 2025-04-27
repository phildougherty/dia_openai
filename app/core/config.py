# app/core/config.py
import os
from pathlib import Path
from typing import List, Optional
import logging
from pydantic import Field, validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

try:
    from dia.model import DEFAULT_SAMPLE_RATE
except ImportError:
    DEFAULT_SAMPLE_RATE = 44100
    logger.warning("Could not import DEFAULT_SAMPLE_RATE from dia.model. Falling back to 44100.")


class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # API settings
    ENABLE_CORS: bool = True
    ALLOWED_ORIGINS: List[str] = ["*"]

    # --- Base Paths ---
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    # Base model directory inside the container (where models are copied in Dockerfile)
    MODEL_DIR: Path = Path("/app/models")
    # Runtime Cache Directory (for DAC model download etc.)
    CACHE_DIR_RUNTIME: Path = Path("/app/cache")
    # --- End Base Paths ---


    # --- Dia Specific Paths (calculated later) ---
    DIA_MODEL_DIR: Optional[Path] = None
    DIA_CONFIG_PATH: Optional[Path] = None
    DIA_CHECKPOINT_PATH: Optional[Path] = None
    # --- End Dia Specific Paths ---

    # Dia Model settings
    USE_TORCH_COMPILE: bool = True
    COMPUTE_DTYPE: str = "float16"

    # Generation Defaults
    OUTPUT_FORMAT: str = "mp3"
    DEFAULT_CFG_SCALE: float = 3.0
    DEFAULT_TEMPERATURE: float = 1.3
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_CFG_FILTER_TOP_K: int = 35
    MIN_AUDIO_LENGTH_SEC: float = 0.1
    MAX_AUDIO_LENGTH_SEC: float = 60
    DIA_DEFAULT_MAX_TOKENS: int = 3072

    # --- App Structure Paths (calculated later) ---
    VOICES_DIR: Optional[Path] = None
    STATIC_DIR: Optional[Path] = None
    # --- End App Structure Paths ---

    @model_validator(mode='after')
    def set_dependent_paths_and_env(cls, values: 'Settings') -> 'Settings':
        """Calculate dependent paths and set runtime environment variables AFTER base paths are set."""
        base_dir = values.BASE_DIR
        cache_dir_runtime = values.CACHE_DIR_RUNTIME
        model_dir = values.MODEL_DIR # Get the already resolved base model dir

        # --- Calculate App Structure Paths ---
        if base_dir:
            if values.VOICES_DIR is None:
                 values.VOICES_DIR = base_dir / "voices"
            if values.STATIC_DIR is None:
                 values.STATIC_DIR = base_dir / "static"
        # --- End Calculate App Structure Paths ---

        # --- Calculate Dia Specific Paths ---
        if model_dir:
            if values.DIA_MODEL_DIR is None:
                values.DIA_MODEL_DIR = model_dir / "dia-1.6b"
            if values.DIA_CONFIG_PATH is None and values.DIA_MODEL_DIR:
                values.DIA_CONFIG_PATH = values.DIA_MODEL_DIR / "config.json"
            if values.DIA_CHECKPOINT_PATH is None and values.DIA_MODEL_DIR:
                values.DIA_CHECKPOINT_PATH = values.DIA_MODEL_DIR / "dia-v0_1.pth"
        # --- End Calculate Dia Specific Paths ---


        # Log the determined paths for debugging
        logger.info(f"Base Dir: {values.BASE_DIR}")
        logger.info(f"Voices Dir (relative to app): {values.VOICES_DIR}")
        logger.info(f"Static Dir (relative to app): {values.STATIC_DIR}")
        logger.info(f"Runtime Cache Dir: {values.CACHE_DIR_RUNTIME}")
        logger.info(f"Base Model Dir (in container): {values.MODEL_DIR}")
        logger.info(f"Dia Model Dir (in container): {values.DIA_MODEL_DIR}")
        logger.info(f"Dia Config Path (in container): {values.DIA_CONFIG_PATH}")
        logger.info(f"Dia Checkpoint Path (in container): {values.DIA_CHECKPOINT_PATH}")

        # Set environment variable for Hugging Face cache at RUNTIME
        if cache_dir_runtime:
            hf_cache_home = cache_dir_runtime / "huggingface"
            os.environ['HF_HOME'] = str(hf_cache_home)
            logger.info(f"Set runtime HF_HOME environment variable to: {os.environ['HF_HOME']}")
        else:
             logger.warning("CACHE_DIR_RUNTIME is not set, using default Hugging Face cache location.")

        # --- Final Validation (Check if paths are valid AFTER calculation) ---
        if not values.DIA_CONFIG_PATH or not values.DIA_CONFIG_PATH.exists():
             logger.error(f"CRITICAL: Dia config file DOES NOT EXIST at calculated path: {values.DIA_CONFIG_PATH}")
             # Consider raising a validation error here to prevent startup
             # raise ValueError(f"Dia config file missing at {values.DIA_CONFIG_PATH}")
        if not values.DIA_CHECKPOINT_PATH or not values.DIA_CHECKPOINT_PATH.exists():
             logger.error(f"CRITICAL: Dia checkpoint file DOES NOT EXIST at calculated path: {values.DIA_CHECKPOINT_PATH}")
             # Consider raising a validation error here
             # raise ValueError(f"Dia checkpoint file missing at {values.DIA_CHECKPOINT_PATH}")
        # --- End Final Validation ---


        return values

    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = 'utf-8'

    # Validators for other fields remain the same
    @validator('COMPUTE_DTYPE')
    def validate_compute_dtype(cls, v):
        allowed_dtypes = ["float16", "bfloat16", "float32"]
        if v not in allowed_dtypes:
            raise ValueError(f"COMPUTE_DTYPE must be one of { allowed_dtypes }")
        return v

    @validator('OUTPUT_FORMAT')
    def validate_output_format(cls, v):
        allowed_formats = ["mp3", "wav", "opus", "flac", "aac", "ogg"]
        if v.lower() not in allowed_formats:
             raise ValueError(f"OUTPUT_FORMAT must be one of { allowed_formats }")
        return v.lower()

    def setup_directories(self):
        """Ensure all required runtime directories exist. Call *after* settings are loaded."""
        # Runtime directories corresponding to volumes or cache
        logger.info("Setting up runtime directories...")
        runtime_dirs_to_create = [
            Path("/app/voices"),      # Corresponds to ./voices volume mount
            Path("/app/static/audio"),# Corresponds to ./static volume mount subdir
            self.CACHE_DIR_RUNTIME / "huggingface" # Runtime cache subdir
        ]
        for dir_path in runtime_dirs_to_create:
             if dir_path: # Check if path is not None
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Ensured runtime directory exists: {dir_path}")
                except OSError as e:
                    logger.error(f"Failed to create runtime directory {dir_path}: {e}")
        # Note: Model directory /app/models/dia-1.6b is created in Dockerfile runner stage

settings = Settings()
# Ensure runtime directories are set up *after* the Settings object is fully initialized
settings.setup_directories() # This now only creates runtime dirs, validation happens in model_validator
