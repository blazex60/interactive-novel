# src/story_engine.py
from __future__ import annotations
import json
import os
from typing import Any, Dict

from story_state import StoryState
from llm_client import call_llm_json

# 環境変数からモデル名を取得（未指定なら gpt-4.1-mini）
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")


def build_world_summary(world: Dict[str, Any]) -> str:
    return json.dumps(world, ensure_ascii=False)


def interpret_input(player_input: str, channel_hint: str) -> Dict[str, Any]:
    """
    自由入力を「どのチャンネルで何をしたいのか」に解釈するフェーズ。
    channel_hint: "chat" or "terminal"
    """
    system_prompt = """
あなたは物語システム用の入力解釈エージェントです。
ユーザーの入力テキストを、次のJSONスキーマに従って解析してください。

出力フォーマット（必ずこの形で返してください。コメントは禁止）:
{
  "channel": "chat" | "terminal" | "mail",
  "target_npc": "friend" | "junior" | "netfriend" | "self_past" | null,
  "intent": "string",
  "content_summary": "string",
  "emotional_tone": "neutral" | "anxious" | "angry" | "sad" | "hopeful" | "guilty" | "confused",
  "mail": {
    "target_time": "string or null",
    "message": "string or null"
  },
  "terminal_command": {
    "name": "string or null",
    "args": "string or null"
  },
  "meta": {
    "is_smalltalk": true | false,
    "is_off_topic": true | false,
    "safety_flag": "none" | "self_harm" | "violence" | "etc",
    "confidence": 0.0
  },
  "raw_text": "string"
}

制約:
- JSON 以外の文字を出力してはいけません。
- 不明な項目は null か適切なデフォルト値を入れてください。
"""
    user_prompt = f"""
入力チャンネルのヒント: {channel_hint}
ユーザー入力: {player_input}
"""

    return call_llm_json(LLM_MODEL, system_prompt, user_prompt)


def generate_scene(
    world: Dict[str, Any],
    state: StoryState,
    interpretation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    解釈結果 + StoryState + 世界観から、次のシーンを生成するフェーズ。
    戻り値は JSON で、UI がそのまま参照できる形式。
    """
    system_prompt = """
あなたはインタラクティブノベルゲームの物語生成AIです。
世界観と現在の状態、およびユーザーの行動解釈に基づき、
次のシーンをJSON形式で生成します。

前提:
- 世界観は現代日本の大学キャンパス。
- 主人公は過去の自分にメールを送れるCLIツールを見つけ、世界線が少しずつズレていきます。
- 主人公だけが、世界線変動前の記憶を保持しています（リーディングシュタイナー相当）。
- NPCは常に「今の世界線」の記憶だけを持ち、過去の世界線の出来事は覚えていません。
- 主人公の内面描写 (protagonist_thought) では、
  世界線のズレへの違和感や、改変前との矛盾に気づく描写を適度に含めてください。

出力スキーマ（必ずこの形で返してください）:
{
  "narration": "string",
  "protagonist_thought": "string",
  "npc_messages": [
    {
      "npc": "friend" | "junior" | "netfriend" | "self_past" | "other",
      "name": "string",
      "text": "string"
    }
  ],
  "terminal_log": ["string"],
  "choices_hint": ["string"],
  "state_diff": {
    "worldline_id": 0,
    "mail_uses": 0,
    "accident_risk": 0,
    "friend_trust": 0,
    "junior_trust": 0,
    "netfriend_intimacy": 0,
    "self_guilt": 0,
    "ending_bias_true": 0,
    "ending_bias_bad": 0,
    "ending_bias_neutral": 0,
    "ending_bias_chaos": 0
  },
  "control": {
    "end_flag": "none" | "normal_ending" | "bad_ending" | "true_ending" | "neutral_ending" | "chaos_ending",
    "ending_type": "none" | "normal" | "bad" | "true" | "neutral" | "chaos",
    "suggested_next_channel": "chat" | "terminal" | "mail" | "free",
    "notes_for_system": "string"
  }
}

制約:
- JSON 以外の文字を出力してはいけません。
- テキスト量は1ターンあたりほどほどにしてください（長編小説にはしない）。
- state_diff の値はすべて「増減量」として整数で返してください（例: 0, 1, -1, 2 等）。
"""
    world_summary = build_world_summary(world)
    state_json = json.dumps(state.to_dict(), ensure_ascii=False)
    interp_json = json.dumps(interpretation, ensure_ascii=False)

    user_prompt = f"""
[世界観 world.json の内容]
{world_summary}

[現在の状態 StoryState (JSON)]
{state_json}

[ユーザー入力の解釈 (JSON)]
{interp_json}

上記を踏まえて、次のシーンを生成してください。
"""

    return call_llm_json(LLM_MODEL, system_prompt, user_prompt)
