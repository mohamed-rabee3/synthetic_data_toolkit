import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import ollama as ollama_pkg
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

from src.utils import strip_thinking

logger = logging.getLogger("pipeline.llm")


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: float = 0
    success: bool = True
    error: str | None = None


class RateLimiter:
    """Sliding-window rate limiter with minimum gap to prevent burst."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.min_gap = 60.0 / max(rpm, 1)
        self.window: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            self.window = [t for t in self.window if t > cutoff]

            if len(self.window) >= self.rpm:
                sleep_until = self.window[0] + 60.0
                wait = sleep_until - now
                if wait > 0:
                    await asyncio.sleep(wait)
            elif self.window:
                gap = now - self.window[-1]
                if gap < self.min_gap:
                    await asyncio.sleep(self.min_gap - gap)

            self.window.append(time.monotonic())


# ---------------------------------------------------------------------------
# Gemini — supports both AI Studio (api_key) and Vertex AI (project/ADC)
# ---------------------------------------------------------------------------


class GeminiClient:
    """
    Google Gemini client via the unified google-genai SDK.

    Modes (set in config or via GEMINI_MODE env var):
      - "ai_studio"  : uses a plain API key (GOOGLE_API_KEY). No GCP project needed.
      - "vertex_ai"  : uses Vertex AI with GCP project + ADC credentials.
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "gemini-2.0-flash-lite")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay_seconds", 5)
        self.rate_limiter = RateLimiter(config.get("requests_per_minute", 500))

        mode = config.get("mode", "vertex_ai")

        if mode == "ai_studio":
            # The SDK reads GOOGLE_GENAI_USE_VERTEXAI from env and overrides
            # client settings, so we must clear it when using AI Studio mode.
            os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

            api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini AI Studio mode requires an API key. "
                    "Set GOOGLE_API_KEY in .env or generation_llm.api_key in pipeline.yaml"
                )
            self.client = genai.Client(api_key=api_key)
            logger.info("Gemini client: AI Studio mode (api_key)")

        elif mode == "vertex_ai":
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

            project = config.get("project") or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = config.get("location") or os.getenv("GOOGLE_CLOUD_LOCATION", "global")

            kwargs: dict[str, Any] = {
                "vertexai": True,
                "http_options": HttpOptions(api_version="v1"),
            }
            if project:
                kwargs["project"] = project
            if location:
                kwargs["location"] = location

            self.client = genai.Client(**kwargs)
            logger.info(
                "Gemini client: Vertex AI mode (project=%s, location=%s)",
                project,
                location,
            )
        else:
            raise ValueError(f"Unknown Gemini mode: '{mode}'. Use 'ai_studio' or 'vertex_ai'.")

    async def call(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        system_parts = []
        contents_parts = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                contents_parts.append(msg["content"])

        contents_text = "\n\n".join(contents_parts)

        gen_config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_parts:
            gen_config.system_instruction = system_parts
        if json_mode:
            gen_config.response_mime_type = "application/json"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                await self.rate_limiter.acquire()
                start = time.monotonic()

                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=contents_text,
                    config=gen_config,
                )

                latency = (time.monotonic() - start) * 1000
                text = response.text or ""

                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ) or 0
                    completion_tokens = getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ) or 0

                logger.debug(
                    "Gemini call: model=%s prompt_tok=%d compl_tok=%d latency=%.0fms",
                    self.model,
                    prompt_tokens,
                    completion_tokens,
                    latency,
                )

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=self.model,
                    latency_ms=latency,
                )

            except Exception as e:
                last_error = str(e)
                is_rate_limit = "429" in last_error or "RESOURCE_EXHAUSTED" in last_error
                logger.warning(
                    "Gemini call failed (attempt %d/%d)%s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    " [rate-limited]" if is_rate_limit else "",
                    last_error,
                )
                if attempt < self.max_retries:
                    base = self.retry_delay * (2**attempt)
                    if is_rate_limit:
                        base = max(base, 10)
                    jitter = random.uniform(0.5, 1.5)
                    await asyncio.sleep(base * jitter)

        return LLMResponse(
            text="",
            model=self.model,
            success=False,
            error=last_error,
        )


# ---------------------------------------------------------------------------
# Ollama — supports both local and cloud (ollama.com hosted)
# ---------------------------------------------------------------------------


class OllamaClient:
    """
    Ollama client supporting local and cloud modes.

    Modes (set in config or via OLLAMA_MODE env var):
      - "local" : connects to a local Ollama instance (OLLAMA_HOST, default localhost:11434)
      - "cloud" : connects to Ollama's cloud API at https://ollama.com using OLLAMA_API_KEY
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "qwen3")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay_seconds", 5)
        self.rate_limiter = RateLimiter(config.get("requests_per_minute", 60))

        mode = config.get("mode", "local")

        if mode == "cloud":
            api_key = config.get("api_key") or os.getenv("OLLAMA_API_KEY")
            if not api_key:
                raise ValueError(
                    "Ollama cloud mode requires an API key. "
                    "Set OLLAMA_API_KEY in .env or judge_llm.api_key in pipeline.yaml"
                )
            host = "https://ollama.com"
            headers = {"Authorization": f"Bearer {api_key}"}
            self.client = ollama_pkg.Client(host=host, headers=headers)
            self.async_client = ollama_pkg.AsyncClient(host=host, headers=headers)
            logger.info("Ollama client: cloud mode (https://ollama.com)")

        elif mode == "local":
            host = config.get("host") or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.client = ollama_pkg.Client(host=host)
            self.async_client = ollama_pkg.AsyncClient(host=host)
            logger.info("Ollama client: local mode (%s)", host)

        else:
            raise ValueError(f"Unknown Ollama mode: '{mode}'. Use 'local' or 'cloud'.")

    async def call(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        ollama_messages = []
        for msg in messages:
            ollama_messages.append(
                {"role": msg["role"], "content": msg["content"]}
            )

        options = {"temperature": temperature, "num_predict": max_tokens}
        fmt = "json" if json_mode else ""

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                await self.rate_limiter.acquire()
                start = time.monotonic()

                response = await self.async_client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    format=fmt or None,
                    options=options,
                )

                latency = (time.monotonic() - start) * 1000
                raw_text = response.get("message", {}).get("content", "")
                text = strip_thinking(raw_text)

                prompt_tokens = response.get("prompt_eval_count", 0) or 0
                completion_tokens = response.get("eval_count", 0) or 0

                logger.debug(
                    "Ollama call: model=%s prompt_tok=%d compl_tok=%d latency=%.0fms",
                    self.model,
                    prompt_tokens,
                    completion_tokens,
                    latency,
                )

                return LLMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=self.model,
                    latency_ms=latency,
                )

            except Exception as e:
                last_error = str(e)
                is_rate_limit = "429" in last_error or "too many" in last_error.lower()
                logger.warning(
                    "Ollama call failed (attempt %d/%d)%s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    " [rate-limited]" if is_rate_limit else "",
                    last_error,
                )
                if attempt < self.max_retries:
                    base = self.retry_delay * (2**attempt)
                    if is_rate_limit:
                        base = max(base, 10)
                    jitter = random.uniform(0.5, 1.5)
                    await asyncio.sleep(base * jitter)

        return LLMResponse(
            text="",
            model=self.model,
            success=False,
            error=last_error,
        )


# ---------------------------------------------------------------------------
# Unified client — routes to the right provider based on config
# ---------------------------------------------------------------------------


class UnifiedLLMClient:
    """Routes calls to Gemini or Ollama based on pipeline config."""

    def __init__(self, pipeline_config: dict):
        self._clients: dict[str, GeminiClient | OllamaClient] = {}
        self._pipeline_config = pipeline_config

        gen_cfg = pipeline_config.get("generation_llm", {})
        if gen_cfg.get("provider") == "gemini":
            self._clients["generation"] = GeminiClient(gen_cfg)
        elif gen_cfg.get("provider") == "ollama":
            self._clients["generation"] = OllamaClient(gen_cfg)

        judge_cfg = pipeline_config.get("judge_llm", {})
        if judge_cfg.get("provider") == "ollama":
            self._clients["judge"] = OllamaClient(judge_cfg)
        elif judge_cfg.get("provider") == "gemini":
            self._clients["judge"] = GeminiClient(judge_cfg)

    def get_client(self, role: str) -> GeminiClient | OllamaClient:
        if role in self._clients:
            return self._clients[role]
        raise ValueError(
            f"No LLM client configured for role '{role}'. "
            f"Available: {list(self._clients.keys())}"
        )

    async def call(
        self,
        role: str,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self.get_client(role)
        kwargs: dict[str, Any] = {"messages": messages, "json_mode": json_mode}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return await client.call(**kwargs)
