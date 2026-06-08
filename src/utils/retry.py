"""
Retry utility for handling transient failures in network operations.

Provides configurable retry logic with exponential backoff for:
- Google Speech-to-Text API calls
- Whisper API calls
- YouTube video downloads
- Any other network-dependent operations
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar

from utils.exceptions import APIError, YouTubeDownloadError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, APIError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    :param max_attempts: Maximum number of retry attempts
    :param base_delay: Initial delay in seconds before first retry
    :param max_delay: Maximum delay in seconds between retries
    :param backoff_factor: Multiplier for delay after each retry
    :param retryable_exceptions: Tuple of exception types that should trigger a retry
    :return: Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            delay = base_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise

                    wait_time = min(delay, max_delay)
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    delay *= backoff_factor

            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            delay = base_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise

                    wait_time = min(delay, max_delay)
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for "
                        f"{func.__name__}: {e}. Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    delay *= backoff_factor

            raise last_exception  # type: ignore[misc]

        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper

    return decorator