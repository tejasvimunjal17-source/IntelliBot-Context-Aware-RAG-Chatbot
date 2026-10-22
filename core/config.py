"""Central configuration for IntelliBot.

Settings come from environment variables. For local use they are read from a
``.env`` file (python-dotenv). On Streamlit Cloud they come from the app's
Secrets. API keys are never hardcoded and never printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_docs"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".txt")
SUPPORTED_PROVIDERS: Tuple[str, ...] = ("openai", "huggingface")

DEFAULT_MAX_UPLOAD_MB = 10

# Chunking limits (Day 2). Size/overlap are measured in characters.
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000
DEFAULT_EMBEDDING_BATCH_SIZE = 32

# Retrieval / vector store (Day 3)
DEFAULT_TOP_K = 4
MAX_TOP_K = 50
# Cosine similarity below this is treated as "not relevant". NOT yet calibrated
# against the real embedding model (it could not be run when this was written);
# tune it with real questions on Days 4 and 7.
DEFAULT_MIN_SIMILARITY = 0.2
DEFAULT_VECTOR_STORE_DIR = "chroma_db"

# LLM answer generation (Day 4)
DEFAULT_LLM_MAX_TOKENS = 500
MAX_LLM_MAX_TOKENS = 4000
DEFAULT_LLM_TIMEOUT_SECONDS = 30
MAX_LLM_TIMEOUT_SECONDS = 300
# Conversation memory (Day 5): how many question+answer exchanges one chat session keeps.
# 0 switches memory off. Older exchanges are dropped.
DEFAULT_CHAT_HISTORY_TURNS = 4
MAX_CHAT_HISTORY_TURNS = 20
# Hugging Face offers an OpenAI-compatible endpoint, so one client library serves both providers.
DEFAULT_HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"

# Values copied straight from .env.example do not count as real keys.
_PLACEHOLDER_MARKERS = ("your-", "your_", "changeme", "xxxx", "<", "paste")


def _clean_secret(value: Optional[str]) -> Optional[str]:
    """Return a usable secret, or None if it is empty or just a placeholder."""
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    if not value:
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return None
    return value


def _read_positive_int(raw: Optional[str], default: int) -> int:
    """Parse a positive integer from text; fall back to ``default`` if invalid."""
    if raw is None:
        return default
    try:
        number = int(str(raw).strip())
    except ValueError:
        return default
    return number if number > 0 else default


def _read_non_negative_int(raw: Optional[str], default: int) -> int:
    """Like _read_positive_int, but 0 is allowed (e.g. zero overlap)."""
    if raw is None:
        return default
    try:
        number = int(str(raw).strip())
    except ValueError:
        return default
    return number if number >= 0 else default


def _read_float_in_range(raw: Optional[str], default: float, low: float, high: float) -> float:
    """Parse a number between low and high (inclusive); fall back to default if invalid."""
    if raw is None:
        return default
    try:
        number = float(str(raw).strip())
    except ValueError:
        return default
    if number != number or not (low <= number <= high):  # NaN or out of range
        return default
    return number


def chunk_settings_are_valid(size: int, overlap: int) -> bool:
    """Size within limits; overlap not negative and at most half the size."""
    return MIN_CHUNK_SIZE <= size <= MAX_CHUNK_SIZE and 0 <= overlap <= size // 2


@dataclass(frozen=True)
class Settings:
    """Application settings. Secret fields are hidden from repr()/logs."""

    llm_provider: str = "openai"
    openai_api_key: Optional[str] = field(default=None, repr=False)
    huggingface_api_token: Optional[str] = field(default=None, repr=False)
    openai_model: str = "gpt-4o-mini"
    huggingface_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    llm_timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS
    huggingface_base_url: str = DEFAULT_HUGGINGFACE_BASE_URL
    chat_history_turns: int = DEFAULT_CHAT_HISTORY_TURNS
    supported_extensions: Tuple[str, ...] = SUPPORTED_EXTENSIONS

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def vector_store_path(self) -> Path:
        """Folder for the persistent index. Relative paths are inside the project."""
        path = Path(self.vector_store_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def active_model(self) -> str:
        return self.huggingface_model if self.llm_provider == "huggingface" else self.openai_model

    @property
    def active_credential(self) -> Optional[str]:
        if self.llm_provider == "huggingface":
            return self.huggingface_api_token
        return self.openai_api_key

    @property
    def has_llm_credentials(self) -> bool:
        return self.active_credential is not None

    def llm_status_message(self) -> str:
        """Safe, human-friendly status. Never contains the key itself."""
        if self.has_llm_credentials:
            return f"LLM provider '{self.llm_provider}': API key found."
        env_name = (
            "HUGGINGFACEHUB_API_TOKEN"
            if self.llm_provider == "huggingface"
            else "OPENAI_API_KEY"
        )
        return (
            f"LLM provider '{self.llm_provider}': no API key found. "
            f"Add {env_name} to your .env file (or Streamlit Secrets)."
        )


def load_settings(
    environ: Optional[Mapping[str, str]] = None,
    env_file: Optional[Path] = None,
) -> Settings:
    """Build Settings from the environment.

    ``environ`` lets tests pass a fake environment. When it is given, no .env
    file is read, so tests never depend on a developer's real secrets.
    """
    if environ is None:
        dotenv_path = env_file or (PROJECT_ROOT / ".env")
        if dotenv_path.exists():
            # override=False: real environment variables win over the file.
            load_dotenv(dotenv_path, override=False)
        environ = os.environ

    provider = (environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        provider = "openai"

    chunk_size = _read_positive_int(environ.get("CHUNK_SIZE"), DEFAULT_CHUNK_SIZE)
    chunk_overlap = _read_non_negative_int(
        environ.get("CHUNK_OVERLAP"), DEFAULT_CHUNK_OVERLAP
    )
    if not chunk_settings_are_valid(chunk_size, chunk_overlap):
        # Invalid pair: use the safe defaults for BOTH values.
        chunk_size, chunk_overlap = DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

    top_k = _read_positive_int(environ.get("TOP_K"), DEFAULT_TOP_K)
    if top_k > MAX_TOP_K:
        top_k = DEFAULT_TOP_K

    llm_max_tokens = _read_positive_int(environ.get("LLM_MAX_TOKENS"), DEFAULT_LLM_MAX_TOKENS)
    if llm_max_tokens > MAX_LLM_MAX_TOKENS:
        llm_max_tokens = DEFAULT_LLM_MAX_TOKENS
    llm_timeout = _read_positive_int(
        environ.get("LLM_TIMEOUT_SECONDS"), DEFAULT_LLM_TIMEOUT_SECONDS
    )
    if llm_timeout > MAX_LLM_TIMEOUT_SECONDS:
        llm_timeout = DEFAULT_LLM_TIMEOUT_SECONDS

    chat_turns = _read_non_negative_int(
        environ.get("CHAT_HISTORY_TURNS"), DEFAULT_CHAT_HISTORY_TURNS
    )
    if chat_turns > MAX_CHAT_HISTORY_TURNS:
        chat_turns = DEFAULT_CHAT_HISTORY_TURNS

    return Settings(
        llm_provider=provider,
        openai_api_key=_clean_secret(environ.get("OPENAI_API_KEY")),
        huggingface_api_token=_clean_secret(environ.get("HUGGINGFACEHUB_API_TOKEN")),
        openai_model=(environ.get("OPENAI_MODEL") or Settings.openai_model).strip(),
        huggingface_model=(
            environ.get("HUGGINGFACE_MODEL") or Settings.huggingface_model
        ).strip(),
        embedding_model=(
            environ.get("EMBEDDING_MODEL") or Settings.embedding_model
        ).strip(),
        max_upload_mb=_read_positive_int(
            environ.get("MAX_UPLOAD_MB"), DEFAULT_MAX_UPLOAD_MB
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_batch_size=_read_positive_int(
            environ.get("EMBEDDING_BATCH_SIZE"), DEFAULT_EMBEDDING_BATCH_SIZE
        ),
        top_k=top_k,
        min_similarity=_read_float_in_range(
            environ.get("MIN_SIMILARITY"), DEFAULT_MIN_SIMILARITY, 0.0, 1.0
        ),
        vector_store_dir=(
            environ.get("VECTOR_STORE_DIR") or DEFAULT_VECTOR_STORE_DIR
        ).strip()
        or DEFAULT_VECTOR_STORE_DIR,
        llm_max_tokens=llm_max_tokens,
        llm_timeout_seconds=llm_timeout,
        huggingface_base_url=(
            environ.get("HUGGINGFACE_BASE_URL") or DEFAULT_HUGGINGFACE_BASE_URL
        ).strip(),
        chat_history_turns=chat_turns,
    )
