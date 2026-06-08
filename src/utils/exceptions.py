"""
Custom exception hierarchy for the audiotext application.

Provides structured error handling with error codes and user-friendly messages,
making debugging easier and improving the user experience.
"""

from typing import Optional


class AudiotextError(Exception):
    """Base exception for all audiotext application errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ) -> None:
        self.error_code = error_code or "UNKNOWN"
        self.original_exception = original_exception
        super().__init__(message)

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "error_code": self.error_code,
            "message": str(self),
        }


class TranscriptionError(AudiotextError):
    """Raised when the transcription process fails."""

    def __init__(
        self,
        message: str = "Transcription failed",
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="TRANSCRIPTION_ERROR",
            original_exception=original_exception,
        )


class ModelLoadError(AudiotextError):
    """Raised when a WhisperX model fails to load."""

    def __init__(
        self,
        model_size: str,
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Failed to load WhisperX model '{model_size}'. "
            f"Check your GPU memory or try a smaller model.",
            error_code="MODEL_LOAD_ERROR",
            original_exception=original_exception,
        )


class AudioProcessingError(AudiotextError):
    """Raised when audio file processing fails."""

    def __init__(
        self,
        file_path: str = "",
        original_exception: Optional[Exception] = None,
    ) -> None:
        path_msg = f" for '{file_path}'" if file_path else ""
        super().__init__(
            message=f"Failed to process audio file{path_msg}. "
            f"The file may be corrupted or in an unsupported format.",
            error_code="AUDIO_PROCESSING_ERROR",
            original_exception=original_exception,
        )


class APIError(AudiotextError):
    """Raised when an external API call fails."""

    def __init__(
        self,
        api_name: str,
        message: str = "API request failed",
        original_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"{api_name}: {message}",
            error_code="API_ERROR",
            original_exception=original_exception,
        )


class YouTubeDownloadError(AudiotextError):
    """Raised when YouTube audio download fails."""

    def __init__(
        self,
        url: str = "",
        original_exception: Optional[Exception] = None,
    ) -> None:
        url_msg = f" from {url}" if url else ""
        super().__init__(
            message=f"Failed to download audio{url_msg}. "
            f"Check the URL or your internet connection.",
            error_code="YOUTUBE_DOWNLOAD_ERROR",
            original_exception=original_exception,
        )


class ValidationError(AudiotextError):
    """Raised when input validation fails."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(
            message=f"Validation failed for '{field}': {reason}",
            error_code="VALIDATION_ERROR",
        )