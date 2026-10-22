"""LLM access for IntelliBot (Day 4).

One small interface (``LLMClient.generate``) and one implementation,
``OpenAICompatibleClient``, which uses the official ``openai`` Python package.
It serves both providers named in the project brief:

* OpenAI            - normal OpenAI endpoint.
* Hugging Face      - Hugging Face's OpenAI-compatible endpoint
                      (``HUGGINGFACE_BASE_URL``, default https://router.huggingface.co/v1).

Safety rules in this file
* API keys come only from settings (environment / .env / Streamlit Secrets).
  They are never logged, printed, or put in an error message.
* Provider error texts are NOT shown to users (they can contain request details).
  Errors are mapped to short, friendly messages by exception *type*.
* A missing key or missing package raises ``LLMUnavailableError`` with setup
  instructions; nothing is faked.

STATUS: written from the public documentation of the ``openai`` package. It has
NOT been run against a live provider (no network / no key in the build
environment). Unit tests use a scripted stand-in client and are labelled MOCK.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, Sequence

from .config import Settings

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLMError(Exception):
    """The LLM call failed. The message is safe to show to the user."""


class LLMUnavailableError(LLMError):
    """The LLM cannot be used at all (no API key, or package not installed)."""


class LLMClient(Protocol):
    """What the answer engine needs from a language model."""

    model: str

    def generate(self, messages: Sequence[Message]) -> str:
        """Send chat messages, return the model's text answer. Raises LLMError."""
        ...


# Friendly messages chosen by exception TYPE NAME (works for the real openai
# exceptions without importing them, and keeps raw provider text out of the UI).
_ERROR_MESSAGES = {
    "AuthenticationError": (
        "The LLM provider rejected the API key. Check that the key in your .env file "
        "(or Streamlit Secrets) is correct and still active."
    ),
    "PermissionDeniedError": (
        "The LLM provider says this key is not allowed to use the chosen model. "
        "Check your account access or choose another model."
    ),
    "RateLimitError": (
        "The LLM provider is limiting requests (rate limit or no credit left). "
        "Wait a moment and try again, and check your account balance."
    ),
    "APITimeoutError": "The LLM provider took too long to answer. Please try again.",
    "Timeout": "The LLM provider took too long to answer. Please try again.",
    "TimeoutError": "The LLM provider took too long to answer. Please try again.",
    "APIConnectionError": (
        "Could not reach the LLM provider. Check your internet connection and try again."
    ),
    "ConnectionError": (
        "Could not reach the LLM provider. Check your internet connection and try again."
    ),
    "NotFoundError": (
        "The LLM provider does not know the chosen model name. "
        "Check OPENAI_MODEL / HUGGINGFACE_MODEL."
    ),
    "BadRequestError": (
        "The LLM provider did not accept the request. Check the model name and settings."
    ),
    "UnprocessableEntityError": (
        "The LLM provider did not accept the request. Check the model name and settings."
    ),
    "InternalServerError": "The LLM provider has a temporary problem. Please try again later.",
}
_GENERIC_MESSAGE = "The LLM provider returned an error. Please try again in a moment."


def friendly_provider_error(exc: BaseException) -> str:
    """Safe user message for an exception raised by the provider library."""
    for cls in type(exc).__mro__:
        if cls.__name__ in _ERROR_MESSAGES:
            return _ERROR_MESSAGES[cls.__name__]
    status = getattr(exc, "status_code", None)
    if status == 401:
        return _ERROR_MESSAGES["AuthenticationError"]
    if status == 403:
        return _ERROR_MESSAGES["PermissionDeniedError"]
    if status == 404:
        return _ERROR_MESSAGES["NotFoundError"]
    if status == 429:
        return _ERROR_MESSAGES["RateLimitError"]
    if isinstance(status, int) and status >= 500:
        return _ERROR_MESSAGES["InternalServerError"]
    return _GENERIC_MESSAGE


class OpenAICompatibleClient:
    """Chat-completions client for OpenAI and OpenAI-compatible endpoints.

    ``sdk_client`` is for tests: any object with ``chat.completions.create``.
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        base_url: Optional[str] = None,
        max_tokens: int = 500,
        timeout_seconds: int = 30,
        sdk_client: Optional[Any] = None,
    ) -> None:
        if not model or not str(model).strip():
            raise LLMUnavailableError("No LLM model name is set. Check OPENAI_MODEL / HUGGINGFACE_MODEL.")
        self.model = str(model).strip()
        self.max_tokens = max_tokens
        self._sdk_client = sdk_client
        if sdk_client is not None:
            return
        if not api_key:
            raise LLMUnavailableError("No API key is set, so answers cannot be generated.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailableError(
                "The 'openai' package is not installed, so answers cannot be generated. "
                "Install it with: pip install openai"
            ) from exc
        try:
            kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": float(timeout_seconds), "max_retries": 1}
            if base_url:
                kwargs["base_url"] = base_url
            self._sdk_client = OpenAI(**kwargs)
        except Exception as exc:  # e.g. invalid base URL
            logger.warning("LLM client could not be created: %s", type(exc).__name__)
            raise LLMUnavailableError(
                "The LLM client could not be set up. Check the provider settings."
            ) from exc

    def generate(self, messages: Sequence[Message]) -> str:
        if not messages:
            raise LLMError("There is nothing to send to the LLM.")
        try:
            response = self._sdk_client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=list(messages),
                temperature=0,  # as factual and repeatable as possible
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            # Log only the exception TYPE: provider messages may contain private details.
            logger.warning("LLM call failed: %s", type(exc).__name__)
            raise LLMError(friendly_provider_error(exc)) from exc

        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError("The LLM provider sent an answer in an unexpected format.") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError("The LLM provider returned an empty answer. Please try again.")
        return text.strip()


def create_llm_client(settings: Settings) -> OpenAICompatibleClient:
    """Build the client for the configured provider.

    Raises LLMUnavailableError (with setup help, never the key) if there is no
    key or the 'openai' package is missing.
    """
    if not settings.has_llm_credentials:
        raise LLMUnavailableError(settings.llm_status_message())
    base_url = settings.huggingface_base_url if settings.llm_provider == "huggingface" else None
    return OpenAICompatibleClient(
        api_key=settings.active_credential,
        model=settings.active_model,
        base_url=base_url,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )
