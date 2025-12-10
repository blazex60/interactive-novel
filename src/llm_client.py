# src/llm_client.py
from __future__ import annotations
import json
import os
from typing import Any, Dict

from openai import AzureOpenAI


def _create_client() -> AzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY が設定されていません。")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT が設定されていません。")

    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


_client = _create_client()


def call_llm_json(model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Azure OpenAI に JSON モードで問い合わせるヘルパー関数。
    `response_format={"type": "json_object"}` を使って JSON 返却を強制する。
    """
    response = _client.chat.completions.create(
        model=model,  # Azure 上のデプロイ名
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    content = response.choices[0].message.content
    return json.loads(content)
