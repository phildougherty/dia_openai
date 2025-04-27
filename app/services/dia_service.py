# app/services/dia_service.py
import os
import time
import base64
import tempfile
import random
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, List

# Third-party libraries
import torch
import numpy as np
import soundfile as sf

# Application-specific imports
from app.core.config import settings

# Set up logging for this module
logger = logging.getLogger(__name__)

# --- Robust Import of Dia Library Components ---
try:
    # Attempt to import the core Dia class and its default sample rate
    from dia.model import Dia, DEFAULT_SAMPLE_RATE
    dia_available = True
    logger.info("Successfully imported Dia library components (Dia, DEFAULT_SAMPLE_RATE).")
except ImportError as e:
    # Log the specific import error for easier debugging
    logger.error(f"Failed to import Dia library components: {e}. DiaService functionality will be unavailable.", exc_info=True)
    # Define placeholders so the rest of the file can be parsed, but functionality will fail
    Dia = None
    DEFAULT_SAMPLE_RATE = 44100 # Fallback sample rate
    dia_available = False
# --- End Robust Import ---


# --- Seeding Helper Function ---
# Based on Dia's cli.py logic for reproducibility
def set_seed(seed: int):
    """Sets the random seed across relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Optional: Force deterministic algorithms for CuDNN. May impact performance.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed globally to: {seed}")
# --- End Seeding Helper ---


# --- Dia Service Singleton Class ---
class DiaService:
    """
    Manages the Dia TTS model lifecycle and provides generation capabilities.
    Uses a singleton pattern to ensure only one instance loads the model.
    Loads the model from local files specified in the application configuration.
    """
    _instance = None
    _lock = threading.Lock() # Lock for thread-safe singleton instantiation

    def __new__(cls, *args, **kwargs):
        """Implements the singleton pattern creation logic."""
        if not cls._instance:
            with cls._lock:
                # Double-check inside lock to prevent race conditions
                if not cls._instance:
                    # Critical check: ensure the Dia library was imported successfully
                    if not dia_available:
                         # Log as critical error as the service cannot function
                         logger.critical("Cannot instantiate DiaService: Dia library failed to import.")
                         raise RuntimeError("Dia library failed to import. Cannot create DiaService.")
                    logger.info("Creating new DiaService instance.")
                    cls._instance = super().__new__(cls)
                    # Add _initialized flag to prevent re-running __init__ logic
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initializes the DiaService, loading the model if not already done."""
        # Prevent re-initialization if instance already exists and is initialized
        if hasattr(self, '_initialized') and self._initialized:
             return

        with self._lock:
            # Check initialization status again inside lock for thread safety
            if hasattr(self, '_initialized') and self._initialized:
                 return

            # Safety check: Ensure Dia class is available before proceeding
            if not Dia:
                logger.critical("Dia class is None during initialization. Import failed earlier.")
                raise RuntimeError("Dia class not available during initialization.")

            self._model: Optional[Dia] = None # Type hint for the model attribute
            try:
                self._load_model() # Load the model during initialization
                self._initialized = True # Mark as initialized *after* successful loading
                logger.info("DiaService initialized successfully and model loaded.")
            except Exception as e:
                 # Ensure initialization fails clearly if model loading fails
                 logger.critical(f"DiaService initialization failed during model loading: {e}", exc_info=True)
                 self._initialized = False # Explicitly mark as not initialized
                 # Re-raise the exception to prevent the service from being used incorrectly
                 raise RuntimeError(f"DiaService could not be initialized due to model loading error: {e}") from e

    def _load_model(self):
        """Loads the Dia model from local files (config + checkpoint) specified in settings."""
        if self._model is not None:
            logger.info("Dia model is already loaded. Skipping reload.")
            return

        logger.info("Loading Dia model from local files...")
        start_time = time.time()

        # Get paths from configuration settings
        config_path = str(settings.DIA_CONFIG_PATH)
        checkpoint_path = str(settings.DIA_CHECKPOINT_PATH)

        # --- Pre-check if model files exist ---
        if not os.path.exists(config_path):
            logger.error(f"Dia configuration file NOT found at expected path: {config_path}")
            raise FileNotFoundError(f"Dia config file missing at: {config_path}")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Dia checkpoint file NOT found at expected path: {checkpoint_path}")
            raise FileNotFoundError(f"Dia checkpoint file missing at: {checkpoint_path}")
        # --- End Pre-check ---

        try:
            compute_dtype = settings.COMPUTE_DTYPE
            logger.info(f"Using compute dtype: {compute_dtype}")
            logger.info(f"Loading config from: {config_path}")
            logger.info(f"Loading checkpoint from: {checkpoint_path}")

            # Use Dia.from_local to load model from files downloaded during Docker build
            self._model = Dia.from_local(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                compute_dtype=compute_dtype
                # device can be explicitly set, e.g., device=torch.device("cuda:0")
                # but Dia usually auto-detects the best available device (GPU > CPU).
            )

            # IMPORTANT: Dia.from_local internally calls _load_dac_model, which downloads
            # the DAC model using dac.utils.download(). This download happens at RUNTIME
            # if the DAC model isn't already present in the Hugging Face cache
            # directory (controlled by HF_HOME environment variable, set in config.py).
            logger.info(f"DAC model required by Dia will be loaded/downloaded to: {os.environ.get('HF_HOME', 'Default HF Cache')}")

            load_time = time.time() - start_time
            logger.info(f"Dia model successfully loaded from local files in {load_time:.2f} seconds.")

        except FileNotFoundError as fnf_err:
             # This should ideally be caught by the pre-check, but handle defensively
             logger.exception(f"FileNotFoundError during Dia.from_local: {fnf_err}")
             self._model = None # Ensure model is None on failure
             raise fnf_err # Re-raise the specific error
        except Exception as e:
            logger.exception("Unexpected error loading Dia model from local files.")
            self._model = None # Ensure model is None on failure
            # Wrap the original exception for better context
            raise RuntimeError(f"Failed to load Dia model from local files: {str(e)}") from e

    def generate_speech(
        self,
        text: str,
        audio_prompt_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        cfg_scale: float = settings.DEFAULT_CFG_SCALE,
        temperature: float = settings.DEFAULT_TEMPERATURE,
        top_p: float = settings.DEFAULT_TOP_P,
        cfg_filter_top_k: int = settings.DEFAULT_CFG_FILTER_TOP_K,
    ) -> Optional[np.ndarray]:
        """
        Generates speech audio from the provided text using the loaded Dia model.

        Args:
            text: The input text to synthesize. Use [S1] and [S2] for speaker turns.
            audio_prompt_bytes: Optional WAV audio data (as bytes) to use for voice cloning.
            seed: Optional integer seed for reproducibility.
            max_tokens: Optional maximum number of audio tokens to generate. Overrides default.
            cfg_scale: Classifier-Free Guidance scale.
            temperature: Sampling temperature for token generation.
            top_p: Nucleus sampling probability.
            cfg_filter_top_k: Top-K filtering for CFG.

        Returns:
            A numpy array containing the generated audio waveform (float32),
            or None if generation fails.

        Raises:
            RuntimeError: If the Dia model is not loaded or generation fails unexpectedly.
            ValueError: If the input text is empty.
            FileNotFoundError: If the temporary audio prompt file handling fails.
        """
        if self._model is None:
            logger.error("Cannot generate speech: Dia model is not loaded.")
            raise RuntimeError("Dia model is not loaded")
        if not text:
            logger.error("Cannot generate speech: Input text cannot be empty.")
            raise ValueError("Text input cannot be empty")

        # Determine whether to use torch.compile based on settings
        use_torch_compile = settings.USE_TORCH_COMPILE if hasattr(torch, 'compile') else False
        # Determine max_tokens, using default from settings if not provided
        max_tokens = max_tokens if max_tokens is not None else settings.DIA_DEFAULT_MAX_TOKENS
        temp_audio_path = None # Path for temporary audio prompt file

        try:
            # Set the seed for this generation run if provided
            if seed is not None:
                set_seed(seed)

            start_time = time.time()

            # --- Handle Audio Prompt ---
            # Dia's generate function expects a file path for the audio prompt.
            # We need to write the provided bytes to a temporary file.
            audio_prompt_path_for_generate = None
            if audio_prompt_bytes:
                # Use mkstemp for a unique, persistent temporary file path
                # We need to manage deletion manually in the finally block.
                try:
                    # Suggest WAV format via suffix, as it's commonly supported
                    fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
                    logger.info(f"Creating temporary file for audio prompt: {temp_audio_path}")
                    # Open the file descriptor in binary write mode and write the bytes
                    with os.fdopen(fd, 'wb') as tmp_file:
                        tmp_file.write(audio_prompt_bytes)
                    # The path is now ready to be passed to Dia
                    audio_prompt_path_for_generate = temp_audio_path
                except Exception as temp_file_error:
                    logger.error(f"Failed to create or write temporary audio prompt file: {temp_file_error}", exc_info=True)
                    # Depending on requirements, either proceed without prompt or raise error
                    # Raising ensures the user knows the prompt wasn't used.
                    raise IOError(f"Failed to handle temporary audio prompt: {temp_file_error}") from temp_file_error
            # --- End Audio Prompt Handling ---

            # Prepare generation parameters dictionary
            generation_params = {
                "text": text,
                "audio_prompt": audio_prompt_path_for_generate,
                "max_tokens": max_tokens,
                "cfg_scale": cfg_scale,
                "temperature": temperature,
                "top_p": top_p,
                "cfg_filter_top_k": cfg_filter_top_k,
                "use_torch_compile": use_torch_compile,
                "verbose": settings.DEBUG, # Link verbosity to debug setting
            }

            # Log key parameters before calling generate
            logger.info(f"Initiating Dia generation: text='{text[:70]}...', "
                        f"prompt={bool(audio_prompt_path_for_generate)}, seed={seed}, max_tokens={max_tokens}, "
                        f"cfg={cfg_scale}, temp={temperature}, top_p={top_p}, top_k={cfg_filter_top_k}, compile={use_torch_compile}")

            # --- Execute Dia Generation ---
            # Use torch.inference_mode() for efficiency and to disable gradient calculations
            with torch.inference_mode():
                 generated_audio_np = self._model.generate(**generation_params)
            # --- End Dia Generation ---

            generation_time = time.time() - start_time

            # Check if generation succeeded
            if generated_audio_np is None:
                # Dia's generate might return None if no valid tokens are produced
                logger.warning(f"Dia generation returned None (no output) for text: {text[:70]}...")
                # Return None to indicate failure in this specific case
                return None

            # Calculate audio duration and Real-Time Factor (RTF)
            sample_rate = self.get_sample_rate()
            audio_length_sec = len(generated_audio_np) / sample_rate if sample_rate > 0 else 0
            rtf = generation_time / audio_length_sec if audio_length_sec > 0 else float('inf')
            logger.info(f"Generated {audio_length_sec:.2f}s audio ({generated_audio_np.shape}) in {generation_time:.2f}s (RTF: {rtf:.2f})")

            return generated_audio_np

        except Exception as e:
            # Catch any unexpected errors during the generation process
            logger.exception(f"Unexpected error during Dia speech generation for text: {text[:70]}...")
            # Re-raise as a runtime error for the API layer to handle
            raise RuntimeError(f"Error generating speech: {str(e)}") from e

        finally:
            # --- Cleanup Temporary File ---
            # Ensure the temporary audio prompt file is deleted regardless of success/failure
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    logger.info(f"Successfully deleted temporary audio prompt file: {temp_audio_path}")
                except OSError as e:
                    # Log error during cleanup but don't interrupt the main flow/error reporting
                    logger.error(f"Error deleting temporary audio file {temp_audio_path}: {e}")
            # --- End Cleanup ---


    def save_audio(self, audio_np: np.ndarray, output_path: Union[str, Path]) -> bool:
        """
        Saves the provided numpy audio array to a file using soundfile.

        Args:
            audio_np: The numpy array containing audio data (expected float32).
            output_path: The path (string or Path object) where the audio file will be saved.
                         The format is determined by the file extension (e.g., .wav, .flac).

        Returns:
            True if saving was successful, False otherwise.
        """
        if self._model is None: # Check if service is operational
            logger.error("Cannot save audio: Dia model (service) is not loaded.")
            return False

        output_path = Path(output_path) # Ensure it's a Path object
        logger.info(f"Attempting to save audio data (shape: {audio_np.shape}, dtype: {audio_np.dtype}) to {output_path}")

        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            sample_rate = self.get_sample_rate() # Get the standard sample rate

            # Basic validation of the numpy array
            if not np.issubdtype(audio_np.dtype, np.number):
                 logger.error(f"Audio data type is not numeric ({audio_np.dtype}), cannot save.")
                 return False

            # Soundfile generally handles float32/int16 well. Warn if float values are large.
            if np.issubdtype(audio_np.dtype, np.floating):
                 max_abs_val = np.max(np.abs(audio_np))
                 if max_abs_val > 1.0:
                     logger.warning(f"Audio data appears to exceed range [-1.0, 1.0] (max abs: {max_abs_val:.2f}). Clipping might occur depending on format.")
                     # Optionally clip here: audio_np = np.clip(audio_np, -1.0, 1.0)

            # Use soundfile to write the audio array
            # Let soundfile infer subtype based on format or use defaults
            sf.write(str(output_path), audio_np, sample_rate)

            logger.info(f"Audio successfully saved to: {output_path}")
            return True

        except sf.LibsndfileError as sfe: # Catch soundfile-specific errors
            logger.error(f"Soundfile error saving audio to {output_path}: {sfe}", exc_info=True)
            return False
        except IOError as ioe: # Catch file system errors
            logger.error(f"IOError saving audio file {output_path}: {ioe}", exc_info=True)
            return False
        except Exception as e: # Catch any other unexpected errors
            logger.exception(f"Unexpected error saving audio to {output_path}")
            return False


    def get_sample_rate(self) -> int:
        """Returns the default sample rate used by the Dia model."""
        # Dia uses a fixed sample rate internally, exposed via DEFAULT_SAMPLE_RATE
        return DEFAULT_SAMPLE_RATE


    def validate_audio_prompt_file(self, audio_file_path: str) -> bool:
        """
        Validates if an audio file can be loaded by Dia's internal mechanism.
        This implicitly checks format and basic integrity.

        Args:
            audio_file_path: Path to the audio file to validate.

        Returns:
            True if the file seems loadable by Dia, False otherwise.
        """
        if self._model is None:
             logger.error("Cannot validate audio: Dia model is not loaded.")
             # Or raise RuntimeError depending on desired behavior
             return False

        file_path = Path(audio_file_path)
        if not file_path.exists():
             logger.error(f"Validation failed: Audio file does not exist at {file_path}")
             return False
        if not file_path.is_file():
            logger.error(f"Validation failed: Path is not a file {file_path}")
            return False

        try:
            # Leverage Dia's own loading function (which uses torchaudio)
            # We don't need the result, just whether it loads without error.
            _ = self._model.load_audio(str(file_path))
            logger.info(f"Audio file validation successful: {file_path}")
            return True
        except RuntimeError as re: # Often indicates torchaudio/libsndfile load errors
             logger.warning(f"Audio file validation failed (RuntimeError, likely format issue): {file_path} - {re}")
             return False
        except Exception as e: # Catch other potential errors during loading
            logger.exception(f"Unexpected error during audio file validation: {file_path}")
            return False

# --- End Dia Service Class ---