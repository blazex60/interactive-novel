# src/main.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from story_state import StoryState, apply_state_diff
from story_engine import interpret_input, generate_scene


WORLD_PATH = Path(__file__).resolve().parent.parent / "world.json"


@st.cache_data
def load_world() -> Dict[str, Any]:
    with open(WORLD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_state() -> StoryState:
    if "story_state" not in st.session_state:
        st.session_state.story_state = StoryState(step=0)
    return st.session_state.story_state


def get_history() -> List[Dict[str, Any]]:
    if "history" not in st.session_state:
        st.session_state.history = []
    return st.session_state.history


def reset_story():
    st.session_state.story_state = StoryState(step=0)
    st.session_state.history = []
    st.session_state.last_scene = None


def render_story_area(scene: Dict[str, Any] | None) -> None:
    st.subheader("ストーリー")
    if scene:
        narration = scene.get("narration") or ""
        thought = scene.get("protagonist_thought") or ""
        if narration:
            st.markdown(narration)
        if thought:
            st.markdown(f"*{thought}*")
    else:
        st.write("物語はまだ始まっていません。入力して世界線を動かしてみてください。")


def render_chat_area(scene: Dict[str, Any] | None) -> None:
    st.subheader("NPCチャット")
    msgs = scene.get("npc_messages", []) if scene else []
    if not msgs:
        st.caption("NPCとの会話はまだありません。")
    for m in msgs:
        npc_name = m.get("name", "???")
        text = m.get("text", "")
        st.markdown(f"**{npc_name}**: {text}")


def render_terminal_area(scene: Dict[str, Any] | None) -> None:
    st.subheader("ターミナルログ")
    logs = scene.get("terminal_log", []) if scene else []
    if not logs:
        st.caption("まだメールツールは動かしていません。")
    for line in logs:
        st.code(line, language="bash")


def render_history_panel(history: List[Dict[str, Any]]) -> None:
    with st.expander("世界線ログ / 履歴"):
        if not history:
            st.caption("まだログはありません。")
        for h in history:
            st.markdown(f"**Step {h.get('step', 0)} | channel: {h.get('channel')}**")
            st.write(h.get("input"))
            scene = h.get("scene")
            if scene:
                wl = h.get("worldline_id", "")
                st.caption(f"worldline: {wl}")
                narration = scene.get("narration")
                if narration:
                    st.markdown(f"> {narration}")
            st.markdown("---")


def handle_player_input(
    player_input: str,
    channel_hint: str,
    world: Dict[str, Any],
    state: StoryState,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # 解釈フェーズ
    interp = interpret_input(player_input, channel_hint)

    # 生成フェーズ
    scene = generate_scene(world, state, interp)

    # state_diff 適用
    diff = scene.get("state_diff", {}) or {}
    apply_state_diff(state, diff)
    state.step += 1

    # ログ更新
    log_entry = {
        "step": state.step,
        "input": player_input,
        "channel": interp.get("channel"),
        "worldline_id": state.worldline_id,
        "scene": scene,
    }
    history.append(log_entry)
    state.append_log(log_entry)

    return scene


def main():
    st.set_page_config(page_title="世界線ノベル（仮）", page_icon="📡", layout="wide")

    world = load_world()
    state = get_state()
    history = get_history()
    last_scene: Dict[str, Any] | None = st.session_state.get("last_scene")

    col_story, col_side = st.columns([2, 1])

    with col_story:
        st.title(world.get("title", "インタラクティブノベル"))

        render_story_area(last_scene)

        st.markdown("### NPCと話す")
        chat_input = st.text_input(
            "メッセージを入力（友人・後輩・ネットの知人など）",
            key="chat_input",
        )

        st.markdown("### ターミナル（メールツール）")
        terminal_input = st.text_input(
            "worldline-tool コマンドを入力",
            key="terminal_input",
            placeholder='例: worldline-tool send --to 2023-05-01T23:14 "明日の飲み会は断れ"',
        )

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            do_submit_chat = st.button("チャットとして送信", type="primary")
        with col_btn2:
            do_submit_terminal = st.button("ターミナルとして実行")
        with col_btn3:
            reset = st.button("世界線をリセット")

        if reset:
            reset_story()
            st.rerun()

        if do_submit_chat and chat_input:
            scene = handle_player_input(
                player_input=chat_input,
                channel_hint="chat",
                world=world,
                state=state,
                history=history,
            )
            st.session_state.last_scene = scene
            st.rerun()

        if do_submit_terminal and terminal_input:
            scene = handle_player_input(
                player_input=terminal_input,
                channel_hint="terminal",
                world=world,
                state=state,
                history=history,
            )
            st.session_state.last_scene = scene
            st.rerun()

    with col_side:
        if last_scene:
            render_chat_area(last_scene)
            render_terminal_area(last_scene)
        render_history_panel(history)

        st.markdown("### 現在のステータス")
        st.json(state.to_dict())


if __name__ == "__main__":
    main()
