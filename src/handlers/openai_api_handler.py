import speech_recognition as sr
import utils.config_manager as cm
from handlers.audio_handler import AudioHandler
from interfaces.transcribable import Transcribable
from models.transcription import Transcription
from openai import OpenAI
from utils.enums import WhisperApiResponseFormats
from utils.env_keys import EnvKeys
from utils.exceptions import APIError
from utils.logger import get_logger
from utils.retry import retry

logger = get_logger(__name__)


class OpenAiApiHandler(Transcribable):
    @staticmethod
    @retry(max_attempts=3, retryable_exceptions=(ConnectionError, TimeoutError, APIError))
    def transcribe(audio_data: sr.AudioData, transcription: Transcription) -> str:
        if not transcription.language_code:
            raise ValueError(
                "The language provided is not correct. Please select one of the list."
            )

        config = cm.ConfigManager.get_config_whisper_api()
        compressed_audio = AudioHandler.compress_audio(audio_data)
        timestamp_granularities = (
            config.timestamp_granularities
            if config.response_format == WhisperApiResponseFormats.VERBOSE_JSON.value
            else None
        )

        client = OpenAI(
            api_key=EnvKeys.OPENAI_API_KEY.get_value(),
            timeout=120.0,  # 2 minutes
        )

        try:
            if timestamp_granularities:
                whisper_api_transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=compressed_audio,
                    language=transcription.language_code,
                    response_format=config.response_format,
                    temperature=config.temperature,
                    timestamp_granularities=timestamp_granularities,
                )
            else:
                whisper_api_transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=compressed_audio,
                    language=transcription.language_code,
                    response_format=config.response_format,
                    temperature=config.temperature,
                )

            result: str
            if WhisperApiResponseFormats.JSON.value in config.response_format:
                result = whisper_api_transcription.to_json()
            else:
                result = str(whisper_api_transcription)

            logger.debug(
                f"Whisper API transcription successful: {len(result)} chars"
            )
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Whisper API transcription failed: {error_msg}")
            raise APIError("OpenAI Whisper API", error_msg)
