import speech_recognition as sr
from interfaces.transcribable import Transcribable
from models.transcription import Transcription
from utils.env_keys import EnvKeys
from utils.exceptions import APIError
from utils.logger import get_logger
from utils.retry import retry

logger = get_logger(__name__)


class GoogleApiHandler(Transcribable):
    @staticmethod
    @retry(max_attempts=3, retryable_exceptions=(ConnectionError, TimeoutError, APIError))
    def transcribe(audio_data: sr.AudioData, transcription: Transcription) -> str:
        r = sr.Recognizer()

        try:
            text = str(
                r.recognize_google(
                    audio_data,
                    language=transcription.language_code,
                    key=EnvKeys.GOOGLE_API_KEY.get_value() or None,
                )
            )
            text = f"{text}. "

            logger.debug(
                f"Google API transcription successful: {len(text)} chars"
            )
            return text

        except sr.UnknownValueError:
            logger.warning("Google API could not understand the audio")
            return ""
        except sr.RequestError as e:
            raise APIError("Google Speech-to-Text", str(e))
