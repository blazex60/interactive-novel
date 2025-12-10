# src/llm_client.py
from __future__ import annotations
import json
import os
from typing import Any, Dict

from openai import OpenAI


def _create_client() -> OpenAI:
    """
    OpenAI API クライアントを作成する。
    - OPENAI_API_KEY 環境変数が必須。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    # 通常の OpenAI エンドポイント
    return OpenAI(api_key=api_key)


_client = _create_client()


def call_llm_json(model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    LLM に JSON オブジェクトだけを返させるユーティリティ。
    """
    resp = _client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )
    content = resp.choices[0].message.content
    return json.loads(content)
