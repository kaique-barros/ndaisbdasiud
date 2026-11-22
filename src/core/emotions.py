# Emoções da persona, seguindo a Roda de Plutchik.
#   - 8 emoções primárias com intensidade em [0, 1] (leve / média / forte mudam o nome:
#     serenidade -> alegria -> êxtase);
#   - díades primárias, secundárias e terciárias (amor, culpa, deleite...) surgem quando as duas
#     emoções componentes estão presentes; intensidade = a menor das duas;
#   - o dicionário de nomes/descrições vem de .configs/engine/emotions_description.json;
#   - decaimento ("erosão"): cada emoção volta ao temperamento (baseline) da persona:
#         x(t) = base + (x0 - base) * 2 ** (-dt / meia_vida)
#     com meia-vida por emoção (surpresa passa rápido, tristeza demora) e dt em tempo real,
#     então depois de uma noite a persona volta ao normal.
#   - emoções opostas se inibem: somar alegria reduz tristeza (opposite_inhibition).
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.utils.config import ConfigError
from src.utils.logger import get_logger

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class Dyad:
    key: str
    label: str
    kind: str  # primary | secondary | tertiary
    intensity: float


class EmotionDictionary:
    """Interpreta e valida o emotions_description.json."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        try:
            self.wheel: list[str] = list(data["wheel"])
            self.primary: dict[str, dict] = dict(data["primary"])
            self.dyads: dict[str, dict] = dict(data["dyads"])
        except KeyError as exc:
            raise ConfigError(f"emotions_description.json sem a chave {exc}") from exc
        if len(self.wheel) != 8 or len(set(self.wheel)) != 8:
            raise ConfigError("'wheel' deve ter as 8 emoções primárias, sem repetição")
        if set(self.primary) != set(self.wheel):
            raise ConfigError("'primary' deve descrever exatamente as emoções de 'wheel'")
        for key, p in self.primary.items():
            if len(p.get("levels", [])) != 3 or not p.get("label"):
                raise ConfigError(f"emoção '{key}' precisa de 'label' e 3 'levels'")
        for key, d in self.dyads.items():
            comps = d.get("components", [])
            if len(comps) != 2 or any(c not in self.primary for c in comps) or comps[0] == comps[1]:
                raise ConfigError(f"díade '{key}' precisa de 2 componentes primárias distintas")
            if not d.get("label"):
                raise ConfigError(f"díade '{key}' sem 'label'")

    def opposite(self, emotion: str) -> str:
        return self.wheel[(self.wheel.index(emotion) + 4) % 8]


@dataclass(frozen=True)
class EmotionParams:
    default_half_life_hours: float = 6.0
    half_life_hours: Mapping[str, float] = None  # type: ignore[assignment]
    opposite_inhibition: float = 0.5
    max_delta: float = 0.5
    dyad_threshold: float = 0.3
    visible_threshold: float = 0.15
    describe_top: int = 3

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "EmotionParams":
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        known["half_life_hours"] = dict(known.get("half_life_hours") or {})
        return cls(**known)

    def half_life(self, emotion: str) -> float:
        return float((self.half_life_hours or {}).get(emotion, self.default_half_life_hours))


class EmotionState:
    def __init__(self, dictionary: EmotionDictionary, params: EmotionParams,
                 baseline: Mapping[str, float] | None = None,
                 state: Mapping[str, float] | None = None,
                 updated_at: datetime | None = None, clock: Clock = utcnow) -> None:
        self.dict, self.params, self.clock = dictionary, params, clock
        self.baseline = {e: _clamp(float((baseline or {}).get(e, 0.0))) for e in dictionary.wheel}
        start = state if state is not None else self.baseline
        self.values = {e: _clamp(float(start.get(e, self.baseline[e]))) for e in dictionary.wheel}
        self.updated_at = updated_at or clock()
        self.log = get_logger("emotions", "core")

    # --- dinâmica ---

    def decay(self, now: datetime | None = None) -> None:
        """Erosão exponencial em direção ao baseline pelo tempo decorrido desde a última atualização."""
        now = now or self.clock()
        hours = (now - self.updated_at).total_seconds() / 3600
        if hours > 0:
            for e in self.values:
                half = max(self.params.half_life(e), 1e-6)
                base = self.baseline[e]
                self.values[e] = base + (self.values[e] - base) * 2 ** (-hours / half)
        self.updated_at = max(now, self.updated_at)

    def apply(self, delta: Mapping[str, float]) -> dict[str, float]:
        """Aplica uma variação (já com decaimento até agora). Devolve a mudança efetiva por
        emoção, incluindo a inibição dos opostos; mudanças menores que 0,01 são omitidas."""
        self.decay()
        before = dict(self.values)
        for emotion, raw in delta.items():
            if emotion not in self.values:
                self.log.warning("emoção desconhecida ignorada: %r", emotion)
                continue
            d = _clamp(float(raw), -self.params.max_delta, self.params.max_delta)
            self.values[emotion] = _clamp(self.values[emotion] + d)
            if d > 0:
                opp = self.dict.opposite(emotion)
                self.values[opp] = _clamp(self.values[opp] - d * self.params.opposite_inhibition)
        net = {e: round(self.values[e] - before[e], 3) for e in self.values}
        return {e: d for e, d in net.items() if abs(d) >= 0.01}

    # --- leitura ---

    def level_name(self, emotion: str, value: float | None = None) -> str:
        v = self.values[emotion] if value is None else value
        levels = self.dict.primary[emotion]["levels"]
        return levels[0 if v < 1 / 3 else 1 if v < 2 / 3 else 2]

    def dyads(self) -> list[Dyad]:
        found = []
        for key, d in self.dict.dyads.items():
            a, b = d["components"]
            level = min(self.values[a], self.values[b])
            if level >= self.params.dyad_threshold:
                found.append(Dyad(key, d["label"], d.get("kind", ""), round(level, 3)))
        return sorted(found, key=lambda x: -x.intensity)

    def active(self) -> list[tuple[str, float]]:
        items = [(e, v) for e, v in self.values.items() if v >= self.params.visible_threshold]
        return sorted(items, key=lambda x: -x[1])

    def describe(self) -> str:
        """Texto para o prompt. Vazio se não há nada marcante."""
        active = self.active()
        if not active:
            return "Calma, sem nenhuma emoção marcante."
        lines = []
        for i, (e, v) in enumerate(active):
            text = f"- {self.level_name(e, v)} ({self.dict.primary[e]['label']}, {v:.2f})"
            if i < self.params.describe_top:
                text += f": {self.dict.primary[e]['description']}"
            lines.append(text)
        for d in self.dyads()[: self.params.describe_top]:
            desc = self.dict.dyads[d.key].get("description", "")
            lines.append(f"- {d.label} ({d.intensity:.2f})" + (f": {desc}" if desc else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"baseline": self.baseline,
                "state": {e: round(v, 4) for e, v in self.values.items()},
                "updated_at": self.updated_at.isoformat(timespec="seconds")}


class EmotionStore:
    """Persistência em .data/personas/<uuid>/emotions.json."""

    def __init__(self, path: Path, dictionary: EmotionDictionary, params: EmotionParams,
                 clock: Clock = utcnow) -> None:
        self.path, self.dict, self.params, self.clock = Path(path), dictionary, params, clock

    def load(self) -> EmotionState:
        data: dict[str, Any] = {}
        if self.path.is_file() and self.path.read_text(encoding="utf-8").strip():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigError(f"JSON inválido em {self.path}: {exc}") from exc
        updated = data.get("updated_at")
        state = EmotionState(self.dict, self.params, data.get("baseline"), data.get("state"),
                             datetime.fromisoformat(updated) if updated else None, self.clock)
        state.decay()  # aplica o tempo em que a persona ficou sem conversar
        return state

    def save(self, state: EmotionState) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.path)  # escrita atômica
