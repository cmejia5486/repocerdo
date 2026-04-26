#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from lib.ai_config import resolve_config, resolved_api_base, resolved_api_key, resolved_api_version, resolved_litellm_model


def exports_for_task(task: str) -> Dict[str, str]:
    cfg = resolve_config(task_name=task or None)
    api_base = resolved_api_base(cfg)
    api_version = resolved_api_version(cfg)
    litellm_model = resolved_litellm_model(cfg)

    out = {
        "AI_TASK": task,
        "AI_PROFILE": str(cfg.get("resolved_profile") or ""),
        "AI_PROVIDER": str(cfg.get("provider") or "openai"),
        "AI_MODEL": str(cfg.get("model") or ""),
        "AI_LITELLM_MODEL": litellm_model,
        "AI_API_BASE": api_base,
        "AI_API_VERSION": api_version,
        "AI_API_KEY_ENV_VAR": str(cfg.get("api_key_env_var") or "OPENAI_API_KEY"),
        "AI_REASONING_EFFORT": str(cfg.get("reasoning_effort") or ""),
        "AI_MAX_OUTPUT_TOKENS": str(cfg.get("max_output_tokens") or ""),
        "AI_BATCH_SIZE": str(cfg.get("batch_size") or ""),
        "AI_TIMEOUT_S": str(cfg.get("timeout_s") or ""),
        "OPENAI_MODEL": str(cfg.get("model") or ""),
        "OPENAI_REASONING_EFFORT": str(cfg.get("reasoning_effort") or ""),
        "OPENAI_MAX_OUTPUT_TOKENS": str(cfg.get("max_output_tokens") or ""),
        "OPENAI_BATCH_SIZE": str(cfg.get("batch_size") or ""),
        "OPENAI_TIMEOUT_S": str(cfg.get("timeout_s") or ""),
    }

    # Do not print secret values; only expose whether the configured secret is present.
    out["AI_KEY_PRESENT"] = "1" if resolved_api_key(cfg) else "0"
    return out


def write_github_env(path: Path, exports: Dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for key, value in exports.items():
            fh.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve provider-agnostic AI config for a workflow task")
    parser.add_argument("--task", required=True)
    parser.add_argument("--github-env", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    exports = exports_for_task(args.task)
    if args.github_env:
        write_github_env(Path(args.github_env), exports)
    if args.json or not args.github_env:
        print(json.dumps(exports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
