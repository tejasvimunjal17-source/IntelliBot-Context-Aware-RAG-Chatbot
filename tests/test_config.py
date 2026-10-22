"""Tests for core/config.py. These use a fake environment (no real secrets)."""

import tempfile
import unittest
from pathlib import Path

from core.config import Settings, load_settings


class SettingsTests(unittest.TestCase):
    def test_defaults_when_environment_is_empty(self):
        s = load_settings(environ={})
        self.assertEqual(s.llm_provider, "openai")
        self.assertEqual(s.max_upload_mb, 10)
        self.assertEqual(s.max_upload_bytes, 10 * 1024 * 1024)
        self.assertFalse(s.has_llm_credentials)

    def test_placeholder_key_from_env_example_is_not_a_real_key(self):
        s = load_settings(environ={"OPENAI_API_KEY": "your-openai-api-key-here"})
        self.assertFalse(s.has_llm_credentials)

    def test_real_looking_key_is_accepted(self):
        s = load_settings(environ={"OPENAI_API_KEY": "sk-test-FAKE-0123456789"})
        self.assertTrue(s.has_llm_credentials)

    def test_blank_and_quoted_values(self):
        self.assertFalse(load_settings(environ={"OPENAI_API_KEY": "   "}).has_llm_credentials)
        s = load_settings(environ={"OPENAI_API_KEY": '"sk-test-FAKE-0123456789"'})
        self.assertEqual(s.openai_api_key, "sk-test-FAKE-0123456789")

    def test_unknown_provider_falls_back_to_openai(self):
        self.assertEqual(load_settings(environ={"LLM_PROVIDER": "banana"}).llm_provider, "openai")

    def test_huggingface_provider_uses_its_own_token(self):
        s = load_settings(
            environ={
                "LLM_PROVIDER": "HuggingFace",
                "OPENAI_API_KEY": "sk-test-FAKE-0123456789",
            }
        )
        self.assertEqual(s.llm_provider, "huggingface")
        self.assertFalse(s.has_llm_credentials)  # OpenAI key does not count
        s2 = load_settings(
            environ={"LLM_PROVIDER": "huggingface", "HUGGINGFACEHUB_API_TOKEN": "hf_FAKEtoken123"}
        )
        self.assertTrue(s2.has_llm_credentials)

    def test_invalid_upload_limit_uses_default(self):
        for bad in ("abc", "0", "-5", ""):
            self.assertEqual(load_settings(environ={"MAX_UPLOAD_MB": bad}).max_upload_mb, 10)
        self.assertEqual(load_settings(environ={"MAX_UPLOAD_MB": "3"}).max_upload_mb, 3)

    def test_secrets_never_appear_in_repr_or_status_message(self):
        secret = "sk-test-FAKE-SECRET-VALUE"
        s = load_settings(environ={"OPENAI_API_KEY": secret})
        self.assertNotIn(secret, repr(s))
        self.assertNotIn(secret, str(s))
        self.assertNotIn(secret, s.llm_status_message())

    def test_missing_key_message_tells_user_what_to_do(self):
        msg = Settings().llm_status_message()
        self.assertIn("OPENAI_API_KEY", msg)


class ChunkSettingsTests(unittest.TestCase):
    def test_chunk_defaults(self):
        s = load_settings(environ={})
        self.assertEqual((s.chunk_size, s.chunk_overlap, s.embedding_batch_size), (600, 100, 32))

    def test_valid_chunk_values_from_environment(self):
        s = load_settings(environ={"CHUNK_SIZE": "800", "CHUNK_OVERLAP": "150", "EMBEDDING_BATCH_SIZE": "8"})
        self.assertEqual((s.chunk_size, s.chunk_overlap, s.embedding_batch_size), (800, 150, 8))

    def test_zero_overlap_is_allowed(self):
        self.assertEqual(load_settings(environ={"CHUNK_OVERLAP": "0"}).chunk_overlap, 0)

    def test_invalid_pair_falls_back_to_defaults_for_both(self):
        for env in (
            {"CHUNK_SIZE": "abc"},
            {"CHUNK_SIZE": "50"},
            {"CHUNK_SIZE": "99999"},
            {"CHUNK_SIZE": "400", "CHUNK_OVERLAP": "300"},   # overlap more than half
            {"CHUNK_OVERLAP": "-3"},
            {"CHUNK_OVERLAP": "x"},
        ):
            with self.subTest(env=env):
                s = load_settings(environ=env)
                self.assertEqual((s.chunk_size, s.chunk_overlap), (600, 100))

    def test_invalid_batch_size_uses_default(self):
        self.assertEqual(load_settings(environ={"EMBEDDING_BATCH_SIZE": "0"}).embedding_batch_size, 32)

    def test_settings_always_make_a_valid_chunking_config(self):
        from core.chunking import ChunkingConfig
        for env in ({}, {"CHUNK_SIZE": "junk"}, {"CHUNK_SIZE": "1000", "CHUNK_OVERLAP": "500"}):
            s = load_settings(environ=env)
            cfg = ChunkingConfig(s.chunk_size, s.chunk_overlap)  # must not raise
            self.assertEqual((cfg.chunk_size, cfg.chunk_overlap), (s.chunk_size, s.chunk_overlap))


class RetrievalSettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = load_settings(environ={})
        self.assertEqual((s.top_k, s.min_similarity, s.vector_store_dir), (4, 0.2, "chroma_db"))

    def test_valid_values(self):
        s = load_settings(environ={"TOP_K": "8", "MIN_SIMILARITY": "0.35", "VECTOR_STORE_DIR": "my_index"})
        self.assertEqual((s.top_k, s.min_similarity, s.vector_store_dir), (8, 0.35, "my_index"))

    def test_boundaries(self):
        self.assertEqual(load_settings(environ={"TOP_K": "50"}).top_k, 50)
        self.assertEqual(load_settings(environ={"MIN_SIMILARITY": "0"}).min_similarity, 0.0)
        self.assertEqual(load_settings(environ={"MIN_SIMILARITY": "1"}).min_similarity, 1.0)

    def test_invalid_values_fall_back_to_defaults(self):
        for env, field, expected in [
            ({"TOP_K": "0"}, "top_k", 4), ({"TOP_K": "-2"}, "top_k", 4),
            ({"TOP_K": "51"}, "top_k", 4), ({"TOP_K": "many"}, "top_k", 4),
            ({"MIN_SIMILARITY": "1.5"}, "min_similarity", 0.2),
            ({"MIN_SIMILARITY": "-0.1"}, "min_similarity", 0.2),
            ({"MIN_SIMILARITY": "nan"}, "min_similarity", 0.2),
            ({"MIN_SIMILARITY": "abc"}, "min_similarity", 0.2),
            ({"VECTOR_STORE_DIR": "   "}, "vector_store_dir", "chroma_db"),
        ]:
            with self.subTest(env=env):
                self.assertEqual(getattr(load_settings(environ=env), field), expected)

    def test_store_path_is_inside_the_project_unless_absolute(self):
        from core.config import PROJECT_ROOT
        self.assertEqual(load_settings(environ={}).vector_store_path, PROJECT_ROOT / "chroma_db")
        absolute = Path(tempfile.gettempdir()) / "somewhere"
        self.assertEqual(load_settings(environ={"VECTOR_STORE_DIR": str(absolute)}).vector_store_path, absolute)


class ChatHistorySettingsTests(unittest.TestCase):
    def test_default(self):
        self.assertEqual(load_settings(environ={}).chat_history_turns, 4)

    def test_valid_values_including_zero_for_memory_off_and_the_maximum(self):
        for raw, expected in (("0", 0), ("1", 1), ("7", 7), ("20", 20), (" 5 ", 5)):
            with self.subTest(raw=raw):
                self.assertEqual(load_settings(environ={"CHAT_HISTORY_TURNS": raw}).chat_history_turns, expected)

    def test_invalid_values_fall_back_to_the_default(self):
        for raw in ("21", "99999", "-1", "many", "", "2.5"):
            with self.subTest(raw=raw):
                self.assertEqual(load_settings(environ={"CHAT_HISTORY_TURNS": raw}).chat_history_turns, 4)


if __name__ == "__main__":
    unittest.main()
