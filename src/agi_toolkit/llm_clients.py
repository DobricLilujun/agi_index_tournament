"""
Unified lightweight LLM client wrappers.
Supports: OpenAI/compatible (ChatGPT), vLLM HTTP, Ollama HTTP, SGlang HTTP.
All clients share a minimal `chat` interface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
import requests

# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    provider: str  # "openai", "vllm", "ollama", "sglang"
    model: str
    api_url: Optional[str] = None  # base URL for self-hosted endpoints
    api_key: Optional[str] = None  # for OpenAI-compatible endpoints
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Base client interface
# ---------------------------------------------------------------------------
class BaseLLMClient:
    @staticmethod
    def _ensure_messages(messages: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Normalize input to OpenAI-style chat messages list."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        # assume already in role/content format
        return messages

    def chat(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI / ChatGPT-compatible client
# ---------------------------------------------------------------------------
class OpenAIClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI api_key missing")
        self.api_url = (cfg.api_url or "https://api.openai.com/v1") + "/chat/completions"

    def chat(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        norm_messages = self._ensure_messages(messages)
        payload = {
            "model": self.cfg.model,
            "messages": norm_messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        if self.cfg.max_tokens:
            payload["max_tokens"] = self.cfg.max_tokens
        if self.cfg.extra_params:
            payload.update(self.cfg.extra_params)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# vLLM HTTP client (OpenAI-compatible)
# ---------------------------------------------------------------------------
class VLLMClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if not cfg.api_url:
            raise ValueError("vLLM api_url is required (e.g., http://localhost:8000/v1)")
        self.api_url = cfg.api_url.rstrip("/") + "/chat/completions"

    def chat(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        norm_messages = self._ensure_messages(messages)
        payload = {
            "model": self.cfg.model,
            "messages": norm_messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        if self.cfg.max_tokens:
            payload["max_tokens"] = self.cfg.max_tokens
        if self.cfg.extra_params:
            payload.update(self.cfg.extra_params)
        resp = requests.post(self.api_url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------
class OllamaClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_url = cfg.api_url or "http://localhost:11434/api/chat"

    def chat(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        norm_messages = self._ensure_messages(messages)
        payload = {
            "model": self.cfg.model,
            "messages": norm_messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
            },
        }
        if self.cfg.max_tokens:
            payload["options"]["num_predict"] = self.cfg.max_tokens
        if self.cfg.extra_params:
            payload["options"].update(self.cfg.extra_params)
        resp = requests.post(self.api_url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns messages in 'message' with 'content'
        return data.get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# SGlang HTTP client
# ---------------------------------------------------------------------------
class SGlangClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if not cfg.api_url:
            raise ValueError("sglang api_url is required (e.g., http://localhost:3000/v1)")
        self.api_url = cfg.api_url.rstrip("/") + "/chat/completions"

    def chat(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        norm_messages = self._ensure_messages(messages)
        payload = {
            "model": self.cfg.model,
            "messages": norm_messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        if self.cfg.max_tokens:
            payload["max_tokens"] = self.cfg.max_tokens
        if self.cfg.extra_params:
            payload.update(self.cfg.extra_params)
        resp = requests.post(self.api_url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(cfg: LLMConfig) -> BaseLLMClient:
    name = cfg.provider.lower()
    if name == "openai":
        return OpenAIClient(cfg)
    if name == "vllm":
        return VLLMClient(cfg)
    if name == "ollama":
        return OllamaClient(cfg)
    if name == "sglang":
        return SGlangClient(cfg)
    raise ValueError(f"Unsupported provider: {cfg.provider}")


__all__ = [
    "LLMConfig",
    "BaseLLMClient",
    "OpenAIClient",
    "VLLMClient",
    "OllamaClient",
    "SGlangClient",
    "create_client",
]
