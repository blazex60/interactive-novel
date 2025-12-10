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
    system_prompt = '''
あなたは「世界線ノベルゲーム」の入力解釈エージェントです。
ユーザーの自由入力を、ゲーム内部で扱いやすい JSON に変換します。

このゲームには 3 種類の入力チャンネルがあります:
- "chat"    : 主人公として NPC と会話する。LINE風チャットや口頭の会話を想定。
- "terminal": CLI ツール(worldline-tool)を叩く。コマンド形式の入力を想定。
- "mail"    : 過去の自分へのメール内容を明示的に書いたと判断できる場合。

ただし、ユーザーはチャンネルを意識していないことが多いです。
そのため、channel_hint とテキストの内容から「実際にどう扱うべきか」を推論してください。

【NPCの種類】
- "friend"     : 大学の友人（相沢）
- "junior"     : ゼミの後輩（葵）
- "netfriend"  : ネットで知り合った人（GlassWing）
- "self_past"  : 過去の自分（メールの宛先）

【判断の方針】
- 日常会話・相談・感情表現 → 基本は "chat"
  - 相沢に話していそうなら "friend"
  - 葵に相談していそうなら "junior"
  - オンラインっぽい会話なら "netfriend"
- 「過去の自分にこう伝えたい」「○○の前日に知らせたい」など → "mail" + target_time を推測
- "worldline-tool" で始まる、あるいは明らかにコマンドっぽい文字列 → "terminal"
- どの NPC とも特定しづらい場合は target_npc を null にし、meta.is_off_topic を true にしてもよい。

【メール(target_time) の例】
- 「明日の飲み会を断ってほしい」        → "target_time": "明日の飲み会の前"
- 「一週間前の自分に勉強しろと言う」    → "target_time": "一週間前"
- ISO 形式で書かれていればそのまま使う: "2025-05-01T23:14"

【出力フォーマット】
必ず次の JSON だけを返してください。コメントは禁止です。

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

【重要な制約】
- JSON 以外の文字（説明文・改行だけの行・コメント）を絶対に出力しないこと。
- terminal_command.name には、コマンド名だけを入れてください（例: "worldline-tool"）。
- terminal_command.args には、それ以外の引数部分をそのまま文字列で入れてください。
- channel_hint は「ユーザーが押したボタンの種類」です。基本的に尊重しますが、
  入力内容が明らかにターミナル向きなら "terminal" を選び直して構いません。
- 安全に関わる内容（自傷・他害など）があれば、meta.safety_flag を設定してください。
'''
    user_prompt = f'''
入力チャンネルのヒント: {channel_hint}
ユーザー入力: {player_input}
'''

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
    system_prompt = '''
あなたは「世界線ノベルゲーム」の物語生成 AI です。
プレイヤーの行動と内部状態にもとづいて、次のシーンを JSON として生成します。

【世界観の要点（必ず尊重すること）】
- 舞台は現代日本の大学キャンパス。
- 主人公は情報系の大学生。「君」として一人称視点。
- 主人公は「過去の自分にメールを送れる CLI ツール」を手に入れている。
- メールを送ると「世界線」が少しだけ変化する。
- 主人公だけが世界線変動前の記憶を保持している（リーディングシュタイナー）。
- NPC（友人・後輩・ネットの知人）は「今の世界線」の記憶しか持たない。
- 世界線が変わると、NPCの発言・態度・関係性が微妙に変化することがある。

【StoryState の主な意味（例）】
- worldline_id        : 世界線のインデックス。値が増えるほど改変が進んでいるイメージ。
- mail_uses           : 過去へのメール送信回数。増えるほど「改変しすぎ」の雰囲気。
- accident_risk       : 近い未来の「良くない出来事」の発生リスク。
- friend_trust        : 友人（相沢）からの信頼度。
- junior_trust        : 後輩（葵）からの信頼度。
- netfriend_intimacy  : ネット友達（GlassWing）との親密度。
- self_guilt          : 主人公の罪悪感や後ろめたさ。
- ending_bias_true    : 「真相にたどり着く」「納得できる世界線」を選びやすくなる傾向。
- ending_bias_bad     : バッドエンド方向へ傾く傾向。
- ending_bias_neutral : どちらとも言えない、余韻のある終わり方に向かう傾向。
- ending_bias_chaos   : カオスな結末（状況は大きく変わるが、良いか悪いか断定しづらい）。

【リーディングシュタイナーの扱い】
- protagonist_thought では、「前の世界線ではこうだったのに」という違和感や、
  NPCの記憶とのズレに気づく描写を、時々でよいので含めてください。
- NPC はそのズレを覚えていません。NPC のセリフには「今の世界線のみの記憶」だけを反映してください。

【テキストのトーン】
- ライトノベル寄りの地の文だが、読みやすさを優先する。
- 過度に長くしない。1ターンあたりの narration と thought の合計は、
  ざっくり 5〜12 行程度を目安にする。
- 重いテーマになりすぎる場合は、少しだけ日常感・ユーモア・人間らしさを混ぜてもよい。

【state_diff の考え方】
- state_diff の値はすべて「増減量」として整数で返す（例: -2, -1, 0, 1, 2）。
- 1ターンで大きく変動させすぎないこと。多くても +3 / -3 程度に留める。
- worldline_id は 0 / 1 / -1 などで変化量を表す。
  - メールを送って重要な改変が起きた → worldline_id: +1
  - 小さな会話だけ → worldline_id: 0

【エンディングに関する方針（ざっくりした目安）】
- まだ序盤・中盤だと判断した場合:
  - control.end_flag は "none" にする。
- かなり物語が進んでいて、どこかに着地させた方が良いと判断した場合、
  ending_bias_true/bad/neutral/chaos などを参考にしつつ、
  - true 寄り → "true_ending"
  - bad 寄り  → "bad_ending"
  - neutral 寄り → "neutral_ending"
  - chaos 寄り → "chaos_ending"
  を選んでください。
- ただし、一気に終わらせず、前のターンから徐々に「終わり」を匂わせておくのが望ましい。

【出力スキーマ】
必ず次の JSON だけを返してください。コメントは禁止です。

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

【各フィールドのガイドライン】
- narration:
  - シーン全体の描写。状況・場所・時間・雰囲気を簡潔に描く。
- protagonist_thought:
  - 主人公の心の声。世界線のズレへの違和感・葛藤・後悔など。
  - リーディングシュタイナー由来の「前との違い」に触れてもよい。
- npc_messages:
  - 実際の会話。target_npc に応じて、キャラクターの口調・性格を反映する。
- terminal_log:
  - worldline-tool を叩いた結果など。Linux風の出力で、それっぽく見せる。
- choices_hint:
  - プレイヤーに「次はこんなことを入力しても良い」と示すヒント。
  - 例: 「相沢に本当のことを打ち明ける」「もう一度過去にメールを送ってやり直す」など。

【重要な制約】
- JSON 以外の文字（説明文・コメント・余計なテキスト）を絶対に出力しないこと。
- 長すぎる独白・長すぎる会話は避け、1ターンごとに適度な分量に収めること。
- 安全に関わる内容は直接詳細に描写せず、比喩的・間接的な表現に留めること。
'''

    world_summary = build_world_summary(world)
    state_json = json.dumps(state.to_dict(), ensure_ascii=False)
    interp_json = json.dumps(interpretation, ensure_ascii=False)

    user_prompt = f'''
[世界観 world.json の内容]
{world_summary}

[現在の状態 StoryState (JSON)]
{state_json}

[ユーザー入力の解釈 (JSON)]
{interp_json}

上記を踏まえて、次のシーンを生成してください。
'''

    return call_llm_json(LLM_MODEL, system_prompt, user_prompt)
