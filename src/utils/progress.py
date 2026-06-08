"""
Progress tracking for audiotext transcription operations.

Provides a callback-based progress tracker that can be used by handlers
to report progress back to the UI.
"""

import logging
import time
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ProgressStage(str, Enum):
    """Stages of the transcription process."""
    LOADING = "Loading model..."
    PROCESSING = "Processing audio..."
    TRANSCRIBING = "Transcribing..."
    ALIGNING = "Aligning subtitles..."
    SAVING = "Saving transcription..."
    COMPLETE = "Complete"
    ERROR = "Error"


class ProgressTracker:
    """
    Tracks progress of transcription operations.

    Provides a mechanism for handlers to report progress via callbacks,
    allowing the UI to display meaningful progress information.
    """

    def __init__(
        self,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """
        :param callback: Function called with (stage_description, percentage).
                         Percentage is 0.0 to 1.0.
        """
        self._callback = callback
        self._current_stage: Optional[ProgressStage] = None
        self._start_time: Optional[float] = None
        self._current_file: Optional[str] = None

    def set_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set or update the progress callback."""
        self._callback = callback

    def start(self, file_name: Optional[str] = None) -> None:
        """Mark the start of an operation."""
        self._start_time = time.time()
        self._current_file = file_name
        self._report(ProgressStage.LOADING, 0.0)

    def update(
        self,
        stage: ProgressStage,
        progress: float,
        detail: Optional[str] = None,
    ) -> None:
        """
        Update progress.

        :param stage: Current stage of the operation
        :param progress: Progress percentage (0.0 to 1.0)
        :param detail: Optional additional detail message
        """
        self._current_stage = stage
        self._report(stage, progress, detail)

    def complete(self) -> None:
        """Mark the operation as complete."""
        elapsed = time.time() - (self._start_time or time.time())
        detail = f"Done in {elapsed:.1f}s"
        self._report(ProgressStage.COMPLETE, 1.0, detail)

    def error(self, message: str) -> None:
        """Report an error."""
        self._report(ProgressStage.ERROR, 0.0, message)

    def _report(
        self,
        stage: ProgressStage,
        progress: float,
        detail: Optional[str] = None,
    ) -> None:
        """Report progress via the callback."""
        file_info = f" [{self._current_file}]" if self._current_file else ""
        detail_info = f" - {detail}" if detail else ""
        message = f"{stage.value}{file_info}{detail_info}"

        if self._callback:
            self._callback(message, progress)

        logger.debug(f"Progress: {progress:.0%} - {message}")


# Global progress tracker instance
progress_tracker = ProgressTracker()