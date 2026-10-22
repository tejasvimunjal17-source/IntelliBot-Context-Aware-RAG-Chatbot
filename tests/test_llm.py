"""Tests for core/llm.py.

  MOCK  - a pretend `openai` package / SDK client. Checks IntelliBot's own handling
          (arguments sent, error mapping, no secret leaks). NOT proof that a real
          provider works.
  LIVE  - a real call to a real provider. Runs ONLY when you set
          RUN_LIVE_LLM_TESTS=1 and a real API key in the environment. Skipped
          otherwise, including in the build environment, so NO live provider has been tested.
"""

import logging
import os
import sys
import types
import unittest
from unittest import mock

from core.config import load_settings
from core.llm import (
    LLMError,
    LLMUnavailableError,
    OpenAICompatibleClient,
    create_llm_client,
    friendly_provider_error,
)

SECRET = "sk-test-FAKE-SECRET-1234567890"


class FakeSdk:
    """MOCK for an OpenAI SDK client: client.chat.completions.create(...)."""

    def __init__(self, content="Hello", raises=None, response=None):
        self.calls = []
        self._content, self._raises, self._response = content, raises, response
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        if self._response is not None:
            return self._response
        message = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def make_error(name, message="provider detail", status_code=None):
    """Create an exception whose class name matches a real openai exception (MOCK)."""
    cls = type(name, (Exception,), {})
    exc = cls(message)
    if status_code is not None:
        exc.status_code = status_code
    return exc


class GenerateTests(unittest.TestCase):
    def client(self, sdk):
        return OpenAICompatibleClient(api_key=None, model="m-1", sdk_client=sdk, max_tokens=123)

    def test_sends_model_messages_zero_temperature_and_token_limit(self):
        sdk = FakeSdk("  The answer  ")
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        self.assertEqual(self.client(sdk).generate(messages), "The answer")
        call = sdk.calls[0]
        self.assertEqual((call["model"], call["temperature"], call["max_tokens"]), ("m-1", 0, 123))
        self.assertEqual(call["messages"], messages)

    def test_no_messages_is_an_error(self):
        with self.assertRaises(LLMError):
            self.client(FakeSdk()).generate([])

    def test_empty_or_missing_content_is_an_error(self):
        for content in ("", "   ", None):
            with self.subTest(content=content):
                with self.assertRaises(LLMError):
                    self.client(FakeSdk(content)).generate([{"role": "user", "content": "x"}])

    def test_unexpected_response_shape_is_an_error(self):
        for bad in (types.SimpleNamespace(choices=[]), types.SimpleNamespace(), "not a response object"):
            with self.subTest(bad=bad):
                with self.assertRaises(LLMError):
                    self.client(FakeSdk(response=bad)).generate([{"role": "user", "content": "x"}])

    def test_blank_model_name_is_rejected(self):
        with self.assertRaises(LLMUnavailableError):
            OpenAICompatibleClient(api_key="k", model="  ", sdk_client=FakeSdk())


class ErrorMappingTests(unittest.TestCase):
    CASES = {
        "AuthenticationError": "API key",
        "PermissionDeniedError": "not allowed",
        "RateLimitError": "rate limit",
        "APITimeoutError": "too long",
        "APIConnectionError": "reach",
        "NotFoundError": "model name",
        "BadRequestError": "did not accept",
        "InternalServerError": "temporary",
    }

    def test_each_provider_error_becomes_a_friendly_message(self):
        for name, expected in self.CASES.items():
            with self.subTest(error=name):
                sdk = FakeSdk(raises=make_error(name, f"raw text with {SECRET}"))
                client = OpenAICompatibleClient(api_key=None, model="m", sdk_client=sdk)
                with self.assertRaises(LLMError) as ctx:
                    client.generate([{"role": "user", "content": "x"}])
                message = str(ctx.exception)
                self.assertIn(expected.lower(), message.lower())
                self.assertNotIn(SECRET, message)
                self.assertNotIn("raw text", message)

    def test_subclasses_of_known_errors_are_recognised(self):
        base = type("RateLimitError", (Exception,), {})
        child = type("MyRateLimit", (base,), {})
        self.assertIn("rate limit", friendly_provider_error(child("x")).lower())

    def test_status_codes_are_used_when_the_type_is_unknown(self):
        for code, expected in ((401, "API key"), (403, "not allowed"), (404, "model name"),
                               (429, "rate limit"), (503, "temporary")):
            with self.subTest(code=code):
                self.assertIn(expected.lower(), friendly_provider_error(make_error("WeirdError", status_code=code)).lower())

    def test_unknown_errors_get_a_generic_message(self):
        message = friendly_provider_error(RuntimeError(f"boom {SECRET}"))
        self.assertNotIn(SECRET, message)
        self.assertNotIn("boom", message)

    def test_logs_contain_the_error_type_but_never_the_error_text(self):
        sdk = FakeSdk(raises=make_error("AuthenticationError", f"bad key {SECRET}"))
        client = OpenAICompatibleClient(api_key=None, model="m", sdk_client=sdk)
        with self.assertLogs("core.llm", level=logging.WARNING) as logs:
            with self.assertRaises(LLMError):
                client.generate([{"role": "user", "content": "x"}])
        joined = "\n".join(logs.output)
        self.assertIn("AuthenticationError", joined)
        self.assertNotIn(SECRET, joined)


class FakeOpenAIPackage:
    """MOCK `openai` module: records how OpenAI(...) was constructed."""

    def __init__(self):
        self.constructed = []
        outer = self

        class OpenAI:
            def __init__(self, **kwargs):
                outer.constructed.append(kwargs)
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: None))

        self.module = types.ModuleType("openai")
        self.module.OpenAI = OpenAI


class FactoryTests(unittest.TestCase):
    def test_no_key_means_unavailable_with_setup_help_and_no_secret(self):
        with self.assertRaises(LLMUnavailableError) as ctx:
            create_llm_client(load_settings(environ={}))
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_placeholder_key_counts_as_no_key(self):
        with self.assertRaises(LLMUnavailableError):
            create_llm_client(load_settings(environ={"OPENAI_API_KEY": "your-openai-api-key-here"}))

    def test_missing_package_gives_install_message(self):
        settings = load_settings(environ={"OPENAI_API_KEY": SECRET})
        with mock.patch.dict(sys.modules, {"openai": None}):
            with self.assertRaises(LLMUnavailableError) as ctx:
                create_llm_client(settings)
        self.assertIn("pip install openai", str(ctx.exception))
        self.assertNotIn(SECRET, str(ctx.exception))

    def test_openai_provider_uses_default_endpoint_and_settings(self):
        fake = FakeOpenAIPackage()
        settings = load_settings(environ={"OPENAI_API_KEY": SECRET, "OPENAI_MODEL": "my-model",
                                          "LLM_MAX_TOKENS": "321", "LLM_TIMEOUT_SECONDS": "12"})
        with mock.patch.dict(sys.modules, {"openai": fake.module}):
            client = create_llm_client(settings)
        kwargs = fake.constructed[0]
        self.assertEqual(kwargs["api_key"], SECRET)
        self.assertNotIn("base_url", kwargs)
        self.assertEqual(kwargs["timeout"], 12.0)
        self.assertEqual((client.model, client.max_tokens), ("my-model", 321))

    def test_huggingface_provider_uses_its_endpoint_and_token(self):
        fake = FakeOpenAIPackage()
        settings = load_settings(environ={"LLM_PROVIDER": "huggingface", "HUGGINGFACEHUB_API_TOKEN": "hf_FAKEtoken123456",
                                          "HUGGINGFACE_MODEL": "org/model", "OPENAI_API_KEY": SECRET})
        with mock.patch.dict(sys.modules, {"openai": fake.module}):
            client = create_llm_client(settings)
        kwargs = fake.constructed[0]
        self.assertEqual(kwargs["api_key"], "hf_FAKEtoken123456")  # not the OpenAI key
        self.assertEqual(kwargs["base_url"], "https://router.huggingface.co/v1")
        self.assertEqual(client.model, "org/model")

    def test_huggingface_without_token_is_unavailable_even_with_openai_key(self):
        settings = load_settings(environ={"LLM_PROVIDER": "huggingface", "OPENAI_API_KEY": SECRET})
        with self.assertRaises(LLMUnavailableError) as ctx:
            create_llm_client(settings)
        self.assertIn("HUGGINGFACEHUB_API_TOKEN", str(ctx.exception))

    def test_client_construction_failure_is_friendly(self):
        fake = FakeOpenAIPackage()
        fake.module.OpenAI = mock.Mock(side_effect=ValueError(f"bad url {SECRET}"))
        with mock.patch.dict(sys.modules, {"openai": fake.module}):
            with self.assertRaises(LLMUnavailableError) as ctx:
                create_llm_client(load_settings(environ={"OPENAI_API_KEY": SECRET}))
        self.assertNotIn(SECRET, str(ctx.exception))

    def test_settings_repr_never_contains_the_key(self):
        self.assertNotIn(SECRET, repr(load_settings(environ={"OPENAI_API_KEY": SECRET})))


@unittest.skipUnless(
    os.environ.get("RUN_LIVE_LLM_TESTS") == "1" and load_settings().has_llm_credentials,
    "LIVE: set RUN_LIVE_LLM_TESTS=1 and a real API key to run. No live provider has been tested.",
)
class LiveProviderTests(unittest.TestCase):
    def test_real_provider_answers_a_trivial_prompt(self):
        client = create_llm_client(load_settings())
        reply = client.generate([{"role": "user", "content": "Reply with the single word: OK"}])
        self.assertTrue(reply)


if __name__ == "__main__":
    unittest.main()
