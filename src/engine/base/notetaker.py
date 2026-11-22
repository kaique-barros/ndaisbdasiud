# O notetaker: gera as notas pessoais da persona (impressões, o que sentiu), que vão para
# memory/notes. É a parte emocional: grandes alterações emocionais geram nota; nem todo
# conhecimento precisa gerar.
#
# Em lote e em dois passos (como pensado no projeto):
#   1. SEM modelo: separa as mensagens em blocos do mesmo contexto (mesma conversa, sem pausa
#      longa) e soma a "alteração total" de cada bloco, mantendo a alteração de cada mensagem;
#   2. COM modelo: só os blocos cuja alteração total passa de `min_significance` são formatados
#      (mensagens + alterações) e enviados juntos, em uma única chamada.
# Se nenhum bloco é significativo, o modelo nem é chamado.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Sequence

from src.core.actor import render
from src.engine.model import Model
from src.memory.flash import Message
from src.utils.jsonx import clip, extract_json_object
from src.utils.logger import get_logger, get_prompt_logger

MAX_NOTE_CHARS = 600
MAX_SUBJECT_CHARS = 80
MAX_MESSAGE_CHARS = 1200


@dataclass(frozen=True)
class Block:
    messages: tuple[Message, ...]
    total: dict[str, float]

    @property
    def significance(self) -> float:
        return round(sum(abs(v) for v in self.total.values()), 3)


@dataclass(frozen=True)
class Note:
    text: str
    subject: str | None
    importance: float
    emotions: dict[str, float] = field(default_factory=dict)


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def split_blocks(messages: Sequence[Message], gap_minutes: float = 30.0) -> list[Block]:
    """Passo 1: blocos por conversa e por pausas longas; soma a variação emocional de cada um."""
    groups: list[list[Message]] = []
    for m in messages:
        if groups:
            prev = groups[-1][-1]
            t0, t1 = _parse_time(prev.created_at), _parse_time(m.created_at)
            long_gap = bool(t0 and t1 and (t1 - t0).total_seconds() > gap_minutes * 60)
            if m.conversation_id != prev.conversation_id or long_gap:
                groups.append([])
        if not groups:
            groups.append([])
        groups[-1].append(m)
    blocks = []
    for group in groups:
        total: dict[str, float] = {}
        for m in group:
            for emotion, delta in m.emotion_delta.items():
                total[emotion] = total.get(emotion, 0.0) + delta
        blocks.append(Block(tuple(group), {e: round(v, 3) for e, v in total.items()
                                           if abs(v) >= 0.01}))
    return blocks


def _fmt_delta(delta: Mapping[str, float], labels: Mapping[str, str]) -> str:
    return ", ".join(f"{labels.get(e, e)} {v:+.2f}"
                     for e, v in sorted(delta.items(), key=lambda kv: -abs(kv[1])))


def format_blocks(blocks: Sequence[Block], persona_name: str, labels: Mapping[str, str]) -> str:
    out = []
    for i, block in enumerate(blocks, 1):
        total = _fmt_delta(block.total, labels) or "nenhuma"
        out.append(f"## Trecho {i} (variação emocional total: {total})")
        for m in block.messages:
            who = "usuário" if m.role == "user" else persona_name
            line = f"{who}: {clip(m.content, MAX_MESSAGE_CHARS)}"
            if m.emotion_delta:
                line += f"  [{_fmt_delta(m.emotion_delta, labels)}]"
            out.append(line)
        out.append("")
    return "\n".join(out).strip()


def parse_notes(raw: str, blocks: Sequence[Block]) -> list[Note]:
    """Valida a saída do modelo. `block` é o número (1..n) do trecho enviado."""
    data = extract_json_object(raw)
    items = data.get("notes")
    if not isinstance(items, list):
        raise ValueError("a resposta do notetaker não tem a lista 'notes'")
    notes: list[Note] = []
    used: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        number, text = item.get("block"), item.get("text")
        if (not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= len(blocks)
                or number in used or not isinstance(text, str) or not text.strip()):
            continue
        used.add(number)  # no máximo uma nota por trecho
        block = blocks[number - 1]
        subject = item.get("subject")
        importance = item.get("importance")
        if not isinstance(importance, (int, float)) or isinstance(importance, bool):
            importance = min(1.0, block.significance)
        notes.append(Note(
            text=clip(text, MAX_NOTE_CHARS),
            subject=clip(subject, MAX_SUBJECT_CHARS) if isinstance(subject, str) and subject.strip() else None,
            importance=round(max(0.0, min(1.0, float(importance))), 2),
            emotions=dict(block.total)))
    return notes


class Notetaker:
    def __init__(self, model: Model, template: str, persona_name: Callable[[], str],
                 persona_text: Callable[[], str], emotion_labels: Mapping[str, str],
                 min_significance: float = 0.3, gap_minutes: float = 30.0) -> None:
        self.model, self.template = model, template
        self.persona_name, self.persona_text = persona_name, persona_text
        self.labels = dict(emotion_labels)
        self.min_significance, self.gap_minutes = min_significance, gap_minutes
        self.log = get_logger("notetaker", "engine")
        self.prompt_log = get_prompt_logger("notetaker", "engine")

    async def take_notes(self, messages: Sequence[Message]) -> list[Note]:
        blocks = [b for b in split_blocks(messages, self.gap_minutes)
                  if b.significance >= self.min_significance]
        if not blocks:
            self.log.info("nenhum trecho significativo em %d mensagens; sem chamada ao modelo",
                          len(messages))
            return []
        name = self.persona_name()
        prompt = render(self.template, {
            "name": name, "persona": self.persona_text(),
            "blocks": format_blocks(blocks, name, self.labels)})
        self.prompt_log.info("%s", prompt)
        response = await self.model.generate([{"role": "system", "content": prompt},
                                              {"role": "user", "content": "Escreva as notas."}])
        notes = parse_notes(response.text, blocks)
        self.log.info("%d trechos enviados, %d notas geradas", len(blocks), len(notes))
        return notes
