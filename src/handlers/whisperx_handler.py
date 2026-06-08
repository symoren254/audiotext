import os
import traceback
from pathlib import Path
from typing import Optional, Union

import utils.config_manager as cm
import whisperx
from models.transcription import Transcription
from utils.exceptions import TranscriptionError
from utils.logger import get_logger
from utils.model_cache import whisperx_cache
from whisperx.types import AlignedTranscriptionResult, TranscriptionResult

logger = get_logger(__name__)


class WhisperXHandler:
    def __init__(self) -> None:
        self._whisperx_result: Optional[
            Union[TranscriptionResult, AlignedTranscriptionResult]
        ] = None

    async def transcribe_file(self, transcription: Transcription) -> str:
        """
        Transcribe audio from a file using the WhisperX library.

        **IMPROVEMENT**: Uses model cache to avoid reloading WhisperX models
        for each transcription. This can reduce loading time from minutes to seconds.

        :param transcription: An instance of Transcription containing information about
                              the audio file.
        :type transcription: Transcription
        :return: The transcribed text or an error message if transcription fails.
        :rtype: str
        """
        if not transcription.output_file_types:
            raise ValueError(
                "No output file types specified. Please make sure to select at least "
                "one."
            )

        config_whisperx = cm.ConfigManager.get_config_whisperx()

        device = "cpu" if config_whisperx.use_cpu else "cuda"
        task = "translate" if transcription.should_translate else "transcribe"

        try:
            # Use the cached model - avoids reloading for every transcription
            model = whisperx_cache.get_model(
                model_size=config_whisperx.model_size,
                device=device,
                compute_type=config_whisperx.compute_type,
                task=task,
                language=transcription.language_code,
            )

            audio_path = str(transcription.audio_source_path)
            audio = whisperx.load_audio(audio_path)

            logger.info(
                f"Starting WhisperX transcription: {transcription.audio_source_path.name} "
                f"(model: {config_whisperx.model_size}, device: {device})"
            )
            self._whisperx_result = model.transcribe(
                audio, batch_size=config_whisperx.batch_size
            )

            if self._whisperx_result is None:
                raise TranscriptionError("Something went wrong while transcribing.")

            text_combined = " ".join(
                segment["text"].strip() for segment in self._whisperx_result["segments"]
            )
            logger.info(
                f"Transcription completed: {len(text_combined)} chars from "
                f"{len(self._whisperx_result['segments'])} segments"
            )

            # Align output if subtitles are needed
            if (
                "srt" in transcription.output_file_types
                or "vtt" in transcription.output_file_types
            ):
                logger.info(
                    f"Aligning transcription for language: {transcription.language_code}"
                )
                model_aligned, metadata = whisperx_cache.get_align_model(
                    language_code=transcription.language_code, device=device
                )
                self._whisperx_result = whisperx.align(
                    self._whisperx_result["segments"],
                    model_aligned,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                logger.info("Alignment completed")

            return text_combined

        except Exception:
            logger.error(
                f"WhisperX transcription failed for {transcription.audio_source_path}",
                exc_info=True,
            )
            return traceback.format_exc()

    def save_transcription(
        self,
        file_path: Path,
        output_file_types: list[str],
        should_overwrite: bool,
    ) -> None:
        """
        Save the transcription as the specified file types.

        :param file_path: The path to the video or audio file for which subtitles are
                          to be generated.
        :type file_path: Path
        :param output_file_types: A list of strings representing the desired output
                                    file types for the generated subtitles. Subtitles
                                    will be generated in each of the specified formats.
        :type output_file_types: list[str]
        :param should_overwrite: Indicates whether existing subtitle files should be
                                overwritten if they exist. If False, subtitles will
                                only be generated if no subtitle file exists for
                                the given format.
        :type should_overwrite: bool
        """
        config_subtitles = cm.ConfigManager.get_config_subtitles()
        output_dir = file_path.parent

        for output_type in output_file_types:
            path_to_check = file_path.parent / f"{file_path.stem}.{output_type}"

            if should_overwrite or not os.path.exists(path_to_check):
                writer = whisperx.transcribe.get_writer(output_type, str(output_dir))

                # https://github.com/m-bain/whisperX/issues/455#issuecomment-1707547704
                if self._whisperx_result:
                    self._whisperx_result["language"] = "en"

                writer(self._whisperx_result, file_path, vars(config_subtitles))
