"""
LLM Interface
=============
Abstraction layer for multiple LLM providers. Supports:

  - Groq   (free tier — llama-3.3-70b-versatile recommended)
  - OpenAI (gpt-4o-mini recommended for cost)
  - Anthropic (claude-haiku recommended for cost)

Provider SDKs are imported lazily so the app still loads even if a
package is not installed. Users only need the package for their chosen provider.

Usage
-----
provider = get_provider("groq", api_key="gsk_...", model="llama-3.3-70b-versatile")
for chunk in provider.stream_chat(messages):
    print(chunk, end="", flush=True)
"""

import time
from abc import ABC, abstractmethod
from typing import Generator, List


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def stream_chat(self, messages: List[dict]) -> Generator[str, None, None]:
        """
        Stream a chat response.

        Parameters
        ----------
        messages : List of {"role": "system"|"user"|"assistant", "content": str}

        Yields
        ------
        String chunks of the response as they arrive.
        """

    def chat(self, messages: List[dict]) -> str:
        """Non-streaming chat — collects all chunks and returns the full response."""
        return "".join(self.stream_chat(messages))

    @abstractmethod
    def validate_key(self) -> bool:
        """Test whether the API key is valid. Returns True on success."""


# ---------------------------------------------------------------------------
# Groq (free tier)
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    """
    Groq free-tier provider.

    Supports llama-3.3-70b-versatile and other open models.
    Free tier has rate limits — set rate_limit_delay_seconds > 0 in settings
    if you see rate limit errors.
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int,
                 rate_limit_delay: float = 2.0):
        super().__init__(api_key, model, temperature, max_tokens)
        self.rate_limit_delay = rate_limit_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "The 'groq' package is not installed. "
                    "Run: pip install groq"
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def stream_chat(self, messages: List[dict]) -> Generator[str, None, None]:
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)

        client = self._get_client()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                yield (
                    "\n\n⚠️ Groq rate limit reached. "
                    "Please wait a moment and try again, or increase the rate limit delay in Settings."
                )
            elif "auth" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                yield "\n\n⚠️ Invalid Groq API key. Please check your key in the Settings page."
            else:
                yield f"\n\n⚠️ Groq error: {error_msg}"

    def validate_key(self) -> bool:
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    OpenAI provider.

    Recommended model: gpt-4o-mini (cost-effective).
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__(api_key, model, temperature, max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is not installed. "
                    "Run: pip install openai"
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def stream_chat(self, messages: List[dict]) -> Generator[str, None, None]:
        client = self._get_client()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            error_msg = str(e)
            if "auth" in error_msg.lower() or "401" in error_msg:
                yield "\n\n⚠️ Invalid OpenAI API key. Please check your key in the Settings page."
            elif "quota" in error_msg.lower() or "429" in error_msg:
                yield "\n\n⚠️ OpenAI quota exceeded. Check your billing at platform.openai.com."
            else:
                yield f"\n\n⚠️ OpenAI error: {error_msg}"

    def validate_key(self) -> bool:
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    Anthropic provider.

    Recommended model: claude-haiku-4-5-20251001 (cost-effective).
    Note: The system message is extracted and passed separately per Anthropic's API.
    """

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__(api_key, model, temperature, max_tokens)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is not installed. "
                    "Run: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def stream_chat(self, messages: List[dict]) -> Generator[str, None, None]:
        client = self._get_client()

        # Anthropic separates the system message from the conversation
        system_prompt = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                conversation.append(msg)

        if not conversation:
            return

        try:
            import anthropic
            with client.messages.stream(
                model=self.model,
                system=system_prompt or "You are a helpful assistant.",
                messages=conversation,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            error_msg = str(e)
            if "auth" in error_msg.lower() or "401" in error_msg:
                yield "\n\n⚠️ Invalid Anthropic API key. Please check your key in the Settings page."
            elif "429" in error_msg:
                yield "\n\n⚠️ Anthropic rate limit reached. Please wait and try again."
            else:
                yield f"\n\n⚠️ Anthropic error: {error_msg}"

    def validate_key(self) -> bool:
        try:
            import anthropic
            client = self._get_client()
            client.messages.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

PROVIDER_MAP = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

PROVIDER_LABELS = {
    "groq": "Groq (Free)",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}


def get_provider(
    name: str,
    api_key: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs,
) -> LLMProvider:
    """
    Create an LLM provider instance.

    Parameters
    ----------
    name        : Provider name — "groq", "openai", or "anthropic".
    api_key     : API key for the provider.
    model       : Model name to use.
    temperature : Sampling temperature (0 = deterministic, 1 = creative).
    max_tokens  : Maximum tokens in the response.
    **kwargs    : Extra provider-specific arguments (e.g. rate_limit_delay for Groq).

    Returns
    -------
    Configured LLMProvider instance.
    """
    name = name.lower()
    cls = PROVIDER_MAP.get(name)
    if cls is None:
        valid = ", ".join(PROVIDER_MAP.keys())
        raise ValueError(f"Unknown provider '{name}'. Valid options: {valid}")

    if name == "groq":
        return cls(api_key, model, temperature, max_tokens,
                   rate_limit_delay=kwargs.get("rate_limit_delay", 2.0))
    return cls(api_key, model, temperature, max_tokens)
