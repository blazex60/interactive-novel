# src/story_state.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List


@dataclass
class StoryState:
    step: int = 0
    worldline_id: str = "α01"
    mail_uses: int = 0
    accident_risk: int = 0
    friend_trust: int = 50
    junior_trust: int = 50
    netfriend_intimacy: int = 50
    self_guilt: int = 0
    ending_bias_true: int = 0
    ending_bias_bad: int = 0
    ending_bias_neutral: int = 0
    ending_bias_chaos: int = 0
    recent_summary: str = ""
    log: List[Dict[str, Any]] = field(default_factory=list)
    # 主人公だけが世界線変動前の記憶を保持する
    reading_steiner: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryState":
        return cls(**data)

    def append_log(self, entry: Dict[str, Any]) -> None:
        self.log.append(entry)
        if len(self.log) > 100:
            self.log = self.log[-100:]


def apply_state_diff(state: StoryState, diff: Dict[str, Any]) -> StoryState:
    """
    LLM が返す state_diff の差分を適用する。
    値はすべて「増減量」を想定（+1, -2 等）。
    """
    numeric_fields = [
        "mail_uses",
        "accident_risk",
        "friend_trust",
        "junior_trust",
        "netfriend_intimacy",
        "self_guilt",
        "ending_bias_true",
        "ending_bias_bad",
        "ending_bias_neutral",
        "ending_bias_chaos",
    ]

    for key in numeric_fields:
        if key in diff:
            current = getattr(state, key)
            delta = int(diff[key])
            setattr(state, key, max(0, current + delta))

    # worldline_id を増加させる（例: α01 → α02）
    if "worldline_id" in diff:
        delta = int(diff["worldline_id"])
        prefix = state.worldline_id[0]
        num = int(state.worldline_id[1:]) + delta
        if num < 1:
            num = 1
        state.worldline_id = f"{prefix}{num:02d}"

    return state
