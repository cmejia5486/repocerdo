#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lib.ai_config import (
    resolve_config,
    resolved_api_base,
    resolved_api_key,
    resolved_api_version,
    resolved_litellm_model,
)

try:
    from litellm import completion  # type: ignore
except Exception:
    completion = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


@dataclass
class AIResponse:
    output_text: str
    raw_response: Any
    output_parsed: Any = None

    def to_dict(self) -> Dict[str, Any]:
        raw = self.raw_response
        if hasattr(raw, "model_dump"):
            try:
                return raw.model_dump()  # type: ignore[attr-defined]
            except Exception:
                pass
        if isinstance(raw, dict):
            return raw
        return {"output_text": self.output_text}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def normalize_messages(input_payload: Any) -> List[Dict[str, str]]:
    if isinstance(input_payload, str):
        return [{"role": "user", "content": input_payload}]
    if isinstance(input_payload, list):
        out: List[Dict[str, str]] = []
        for item in input_payload:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            content = _content_to_text(item.get("content"))
            out.append({"role": role, "content": content})
        return out
    if isinstance(input_payload, dict):
        role = str(input_payload.get("role") or "user")
        content = _content_to_text(input_payload.get("content"))
        return [{"role": role, "content": content}]
    return [{"role": "user", "content": str(input_payload)}]


def extract_output_text(resp: Any) -> str:
    if hasattr(resp, "choices"):
        try:
            choices = getattr(resp, "choices")
            if choices:
                msg = choices[0].message
                content = getattr(msg, "content", "")
                return _content_to_text(content).strip()
        except Exception:
            pass
    if isinstance(resp, dict):
        try:
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                return _content_to_text(msg.get("content")).strip()
        except Exception:
            pass
    return ""


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned empty text")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON output root must be an object")
    return obj


class AIRuntime:
    def __init__(
        self,
        task: Optional[str] = None,
        profile: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.config = resolve_config(
            task_name=task or os.getenv("AI_TASK", "").strip() or None,
            profile_name=profile,
        )
        self.provider = str(self.config.get("provider") or "openai")
        self.model = resolved_litellm_model(self.config)
        self.api_key = resolved_api_key(self.config, explicit_api_key=api_key)
        self.api_base = (api_base or resolved_api_base(self.config) or "").strip()
        self.api_version = resolved_api_version(self.config)
        self.timeout_s = int(self.config.get("timeout_s") or 30)

    def _use_direct_openai(self) -> bool:
        return self.provider in {"openai", "openai_compatible"} and OpenAI is not None

    def available(self) -> bool:
        if self._use_direct_openai():
            return bool(self.api_key)
        return completion is not None and bool(self.api_key)

    def _completion_kwargs(
        self,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "timeout": self.timeout_s,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_version:
            kwargs["api_version"] = self.api_version

        effort = ""
        if reasoning and isinstance(reasoning, dict):
            effort = str(reasoning.get("effort") or "").strip()
        if not effort:
            effort = str(self.config.get("reasoning_effort") or "").strip()

        if max_output_tokens is None:
            max_output_tokens = int(self.config.get("max_output_tokens") or 0) or None
        if max_output_tokens:
            kwargs["max_tokens"] = int(max_output_tokens)

        temperature = self.config.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = temperature

        if effort and self.provider in {"openai", "azure", "anthropic", "gemini", "mistral", "openrouter"}:
            kwargs["reasoning_effort"] = effort

        return kwargs

    def _create_via_openai_sdk(
        self,
        input: Any,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> AIResponse:
        if OpenAI is None:
            raise RuntimeError("openai SDK is not installed")

        if not self.api_key:
            raise RuntimeError(f"Missing API key in env var: {self.config.get('api_key_env_var')}")

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        client = OpenAI(**client_kwargs)

        messages = normalize_messages(input)
        kwargs: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "timeout": self.timeout_s,
        }

        if max_output_tokens is None:
            max_output_tokens = int(self.config.get("max_output_tokens") or 0) or None
        if max_output_tokens:
            kwargs["max_tokens"] = int(max_output_tokens)

        temperature = self.config.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = client.chat.completions.create(**kwargs)
        return AIResponse(output_text=extract_output_text(resp), raw_response=resp)

    def create(
        self,
        input: Any,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> AIResponse:
        if self._use_direct_openai():
            return self._create_via_openai_sdk(
                input=input,
                model=model,
                max_output_tokens=max_output_tokens,
            )

        if completion is None:
            raise RuntimeError("litellm is not installed; install dependencies from requirements/ai-runtime.txt")
        if not self.api_key:
            raise RuntimeError(f"Missing API key in env var: {self.config.get('api_key_env_var')}")

        messages = normalize_messages(input)
        kwargs = self._completion_kwargs(max_output_tokens=max_output_tokens, reasoning=reasoning)
        if model:
            kwargs["model"] = model
        resp = completion(messages=messages, **kwargs)
        return AIResponse(output_text=extract_output_text(resp), raw_response=resp)

    def parse(
        self,
        input: Any,
        text_format: Any,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> AIResponse:
        resp = self.create(
            input=input,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
        )
        obj = extract_json_object(resp.output_text)
        parsed = None
        if text_format is not None:
            if hasattr(text_format, "model_validate"):
                parsed = text_format.model_validate(obj)
            elif hasattr(text_format, "parse_obj"):
                parsed = text_format.parse_obj(obj)
            else:
                parsed = obj
        return AIResponse(output_text=resp.output_text, raw_response=resp.raw_response, output_parsed=parsed)