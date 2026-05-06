#!/usr/bin/env python3
"""Minimal connectivity check for ANTHROPIC_FOUNDRY_* settings.

This script intentionally depends only on the Python standard library.
It does NOT print the API key.

Supported endpoint styles:
    1) Anthropic Messages API compatible endpoint (default):
         - https://api.anthropic.com/v1/messages
    2) OpenAI-compatible chat completions endpoint (common for Azure OpenAI / Azure AI Foundry):
         - https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=...
         - https://{resource}.inference.ai.azure.com/v1/chat/completions

Env vars:
    - ANTHROPIC_FOUNDRY_API_KEY (required)
    - ANTHROPIC_FOUNDRY_ENDPOINT (optional)
    - ANTHROPIC_FOUNDRY_MODEL (optional; used for Anthropic-style endpoints, and OpenAI non-Azure endpoints)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
import urllib.error
import urllib.request


DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-3-5-haiku-latest"


def _is_azure_endpoint(endpoint: str) -> bool:
    lowered = endpoint.lower()
    return "openai.azure.com" in lowered or "cognitiveservices.azure.com" in lowered or "inference.ai.azure.com" in lowered


def _looks_like_anthropic_messages(endpoint: str) -> bool:
    return "/v1/messages" in endpoint.lower()


def _looks_like_openai_chat_completions(endpoint: str) -> bool:
    lowered = endpoint.lower()
    return "/chat/completions" in lowered or lowered.rstrip("/").endswith("/v1/chat/completions")


def _build_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }


def _build_openai_payload(prompt: str, max_tokens: int, *, model: str | None) -> dict:
    payload: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model
    return payload


def _request(endpoint: str, headers: dict[str, str], payload: dict) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    merged_headers = {"content-type": "application/json"}
    merged_headers.update(headers)
    req = urllib.request.Request(endpoint, data=data, method="POST", headers=merged_headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Anthropic/Foundry Messages API connectivity.")
    parser.add_argument(
        "--prompt",
        default="Reply with a single word: OK",
        help="Prompt to send.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("ANTHROPIC_FOUNDRY_ENDPOINT", DEFAULT_ENDPOINT),
        help="Messages API endpoint URL (overrides env var).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ANTHROPIC_FOUNDRY_MODEL", DEFAULT_MODEL),
        help="Model name (overrides env var).",
    )
    parser.add_argument(
        "--kind",
        choices=["auto", "anthropic", "openai"],
        default="auto",
        help="Request style: 'anthropic' for /v1/messages, 'openai' for /chat/completions.",
    )
    return parser.parse_args()


def main() -> int:
    api_key = os.getenv("ANTHROPIC_FOUNDRY_API_KEY", "").strip()
    if not api_key:
        print(
            "Missing ANTHROPIC_FOUNDRY_API_KEY.\n"
            "- If using direnv: ensure you ran `direnv allow` in this repo.\n"
            "- Or set it temporarily: export ANTHROPIC_FOUNDRY_API_KEY=...",
            file=sys.stderr,
        )
        return 2

    args = parse_args()
    args.endpoint = args.endpoint.strip()
    args.model = args.model.strip()

    kind = args.kind
    if kind == "auto":
        if _looks_like_anthropic_messages(args.endpoint) and not _looks_like_openai_chat_completions(args.endpoint):
            kind = "anthropic"
        elif _looks_like_openai_chat_completions(args.endpoint):
            kind = "openai"
        else:
            # Default to Anthropic; user can override with --kind openai
            kind = "anthropic"

    if kind == "anthropic":
        payload = _build_payload(args.model, args.prompt, args.max_tokens)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "x-request-id": str(uuid.uuid4()),
        }
    else:
        # OpenAI-compatible chat completions.
        # Azure endpoints generally use `api-key` and the deployment is encoded in the URL.
        azure = _is_azure_endpoint(args.endpoint)
        headers = {"x-request-id": str(uuid.uuid4())}
        if azure:
            headers["api-key"] = api_key
            model_for_payload = None
        else:
            headers["authorization"] = f"Bearer {api_key}"
            model_for_payload = args.model
        payload = _build_openai_payload(args.prompt, args.max_tokens, model=model_for_payload)

    try:
        status, body = _request(args.endpoint, headers, payload)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        print(f"HTTP {exc.code} calling {args.endpoint}", file=sys.stderr)
        print(err_body[:2000], file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error calling {args.endpoint}: {exc}", file=sys.stderr)
        return 1

    print(f"Success: HTTP {status} from {args.endpoint}")
    # Print a small snippet for sanity (avoid huge dumps)
    print(body[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
