"""
Input validation utilities for audiotext.

Provides centralized validation for user inputs and configuration values.
"""

import re
from pathlib import Path
from typing import Optional

from utils import constants as c
from utils.enums import AudioSource, TranscriptionMethod
from utils.exceptions import ValidationError


def validate_file_path(file_path: Path) -> Path:
    """
    Validate that a file path exists and is a supported format.

    :param file_path: Path to validate
    :return: The validated path
    :raises ValidationError: If the path is invalid
    """
    if not file_path:
        raise ValidationError("file_path", "Path is empty")

    if not file_path.is_file():
        raise ValidationError("file_path", f"File does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in c.SUPPORTED_FILE_EXTENSIONS:
        raise ValidationError(
            "file_path",
            f"Unsupported file type '{suffix}'. Supported: {', '.join(c.SUPPORTED_FILE_EXTENSIONS)}",
        )

    return file_path


def validate_directory_path(dir_path: Path) -> Path:
    """
    Validate that a directory path exists.

    :param dir_path: Path to validate
    :return: The validated path
    :raises ValidationError: If the path is invalid
    """
    if not dir_path:
        raise ValidationError("directory_path", "Path is empty")

    if not dir_path.is_dir():
        raise ValidationError("directory_path", f"Directory does not exist: {dir_path}")

    return dir_path


def validate_youtube_url(url: str) -> str:
    """
    Validate a YouTube URL.

    :param url: YouTube URL to validate
    :return: The validated URL
    :raises ValidationError: If the URL is invalid
    """
    if not url:
        raise ValidationError("youtube_url", "URL is empty")

    # Common YouTube URL patterns
    patterns = [
        r"^https?://(?:www\.)?youtube\.com/watch\?v=.+",
        r"^https?://(?:www\.)?youtu\.be/.+",
        r"^https?://(?:www\.)?youtube\.com/embed/.+",
        r"^https?://(?:www\.)?youtube\.com/shorts/.+",
    ]

    if not any(re.match(pattern, url) for pattern in patterns):
        raise ValidationError(
            "youtube_url",
            "Invalid YouTube URL format. Expected format: "
            "https://www.youtube.com/watch?v=... or https://youtu.be/...",
        )

    return url


def validate_output_file_types(file_types: Optional[list[str]]) -> list[str]:
    """
    Validate that at least one output file type is selected.

    :param file_types: List of output file type extensions
    :return: The validated list
    :raises ValidationError: If no file types selected
    """
    if not file_types:
        raise ValidationError(
            "output_file_types",
            "No output file types selected. Please select at least one.",
        )

    return file_types


def validate_temperature(temperature: float) -> float:
    """
    Validate temperature is in range [0, 1].

    :param temperature: Temperature value
    :return: Validated temperature
    :raises ValidationError: If out of range
    """
    if not 0 <= temperature <= 1:
        raise ValidationError("temperature", "Must be between 0 and 1")
    return temperature


def validate_batch_size(batch_size: int) -> int:
    """
    Validate batch size is positive and within reasonable limits.

    :param batch_size: Batch size value
    :return: Validated batch size
    :raises ValidationError: If invalid
    """
    if batch_size < 1:
        raise ValidationError("batch_size", "Must be at least 1")
    if batch_size > 64:
        raise ValidationError("batch_size", "Maximum recommended value is 64")
    return batch_size


def validate_transcription_input(
    transcription: "Transcription",  # type: ignore[name-defined]  # noqa: F821
) -> None:
    """
    Validate all transcription inputs before processing.

    :param transcription: The Transcription object to validate
    :raises ValidationError: If any validation fails
    """
    # Validate output file types
    validate_output_file_types(transcription.output_file_types)

    # Validate source-specific inputs
    if transcription.audio_source == AudioSource.FILE:
        if transcription.audio_source_path:
            validate_file_path(transcription.audio_source_path)
    elif transcription.audio_source == AudioSource.DIRECTORY:
        if transcription.audio_source_path:
            validate_directory_path(transcription.audio_source_path)
    elif transcription.audio_source == AudioSource.YOUTUBE:
        if transcription.youtube_url:
            validate_youtube_url(transcription.youtube_url)