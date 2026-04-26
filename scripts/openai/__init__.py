#!/usr/bin/env python3
"""
Compatibility shim for scripts that currently do `from openai import OpenAI`.

This local package is intentionally placed under `scripts/openai/` so that when a
workflow runs `python scripts/<tool>.py`, Python resolves this shim before the
third-party `openai` package. The shim delegates calls to LiteLLM and reads all
provider/model settings from `parameters/ai.config.json`.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from lib.ai_runtime import AIRuntime


class _ResponsesAPI:
    def __init__(self, runtime: AIRuntime) -> None:
        self._runtime = runtime

    def create(self, model: Optional[str] = None, input: Any = None, max_output_tokens: Optional[int] = None, reasoning: Optional[dict] = None, **_: Any):
        return self._runtime.create(input=input, model=model, max_output_tokens=max_output_tokens, reasoning=reasoning)

    def parse(self, model: Optional[str] = None, input: Any = None, text_format: Any = None, max_output_tokens: Optional[int] = None, reasoning: Optional[dict] = None, **_: Any):
        return self._runtime.parse(input=input, text_format=text_format, model=model, max_output_tokens=max_output_tokens, reasoning=reasoning)


class OpenAI:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **_: Any) -> None:
        task = os.getenv("AI_TASK", "").strip() or None
        self._runtime = AIRuntime(task=task, api_key=api_key, api_base=base_url)
        self.responses = _ResponsesAPI(self._runtime)
