"""
WhisperX model cache to avoid reloading models for every transcription.

Loading WhisperX models is expensive (can take minutes). This cache keeps
loaded models in memory and reuses them when the same model configuration is requested.
"""

import logging
from typing import Any, Optional

from utils.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class WhisperXModelCache:
    """
    Singleton cache for WhisperX models.

    Caches models keyed by (model_size, device, compute_type, task, language).
    This avoids expensive reloading when transcribing multiple files with the same config.
    """

    _instance: Optional["WhisperXModelCache"] = None
    _models: dict[str, Any] = {}
    _align_models: dict[str, Any] = {}

    def __new__(cls) -> "WhisperXModelCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._models = {}
            cls._align_models = {}
        return cls._instance

    @staticmethod
    def _make_key(
        model_size: str,
        device: str,
        compute_type: str,
        task: str,
        language: Optional[str] = None,
    ) -> str:
        return f"{model_size}|{device}|{compute_type}|{task}|{language or 'auto'}"

    def get_model(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        task: str,
        language: Optional[str] = None,
    ) -> Any:
        """
        Get or load a WhisperX model.

        :param model_size: Model size (tiny, base, small, medium, large-v2, etc.)
        :param device: Device to run on (cpu, cuda)
        :param compute_type: Compute type (int8, float16, float32)
        :param task: Task (transcribe, translate)
        :param language: Language code (optional)
        :return: Loaded WhisperX model
        :raises ModelLoadError: If the model fails to load
        """
        import whisperx

        key = self._make_key(model_size, device, compute_type, task, language)

        if key in self._models:
            logger.info(f"Using cached WhisperX model: {model_size} (key: {key})")
            return self._models[key]

        logger.info(
            f"Loading WhisperX model: {model_size} on {device} "
            f"(compute: {compute_type}, task: {task}, lang: {language or 'auto'})"
        )
        try:
            model = whisperx.load_model(
                model_size,
                device,
                compute_type=compute_type,
                task=task,
                language=language,
            )
            self._models[key] = model
            logger.info(f"Successfully loaded model: {model_size}")
            return model
        except Exception as e:
            raise ModelLoadError(model_size=model_size, original_exception=e)

    def get_align_model(
        self,
        language_code: str,
        device: str,
    ) -> Any:
        """
        Get or load a WhisperX alignment model.

        :param language_code: Language code for alignment
        :param device: Device to run on
        :return: Tuple of (model, metadata)
        """
        import whisperx

        key = f"align|{language_code}|{device}"

        if key in self._align_models:
            logger.info(f"Using cached align model for: {language_code}")
            return self._align_models[key]

        logger.info(f"Loading align model for: {language_code}")
        model, metadata = whisperx.load_align_model(
            language_code=language_code, device=device
        )
        self._align_models[key] = (model, metadata)
        logger.info(f"Successfully loaded align model for: {language_code}")
        return model, metadata

    def clear(self) -> None:
        """Clear all cached models."""
        model_count = len(self._models)
        align_count = len(self._align_models)
        self._models.clear()
        self._align_models.clear()
        logger.info(f"Cleared {model_count} ASR models and {align_count} align models")


# Global cache instance
whisperx_cache = WhisperXModelCache()