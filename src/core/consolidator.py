# Relação com engine/base/consolidator.py igual à de core/actor com engine/main/actor:
# aqui se monta o prompt (lote de mensagens + memórias parecidas, buscadas em paralelo), se valida
# a resposta e se aplicam os resultados em events/knowledge. Prompt e eventos em logs separados.
#
# Salvaguardas:
#   - as buscas por memórias parecidas não contam como "acesso" (touch=False);
#   - `obsolete` só pode apagar ids que foram mostrados ao modelo no prompt;
#   - um fato recém-adicionado nunca é apagado pelo `obsolete` do mesmo lote (reforçar um fato
#     igual devolve o mesmo id).
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from src.core.actor import render
from src.engine.base.consolidator import ConsolidatorEngine
from src.memory.events import EventsMemory
from src.memory.flash import Message
from src.memory.knowledge import KnowledgeMemory
from src.utils.jsonx import clip, extract_json_object
from src.utils.logger import get_logger, get_prompt_logger

MAX_EVENTS, MAX_FACTS = 10, 15
MAX_EVENT_CHARS, MAX_TERM_CHARS = 500, 100
MAX_MESSAGE_CHARS = 1200
QUERY_CHARS = 1000
_ID = re.compile(r"^([EF])(\d+)$")


@dataclass(frozen=True)
class FactSpec:
    subject: str
    relation: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None


@dataclass(frozen=True)
class Consolidation:
    events: list[str] = field(default_factory=list)
    facts: list[FactSpec] = field(default_factory=list)
    obsolete: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConsolidationResult:
    events_added: int = 0
    facts_added: int = 0
    removed: int = 0


def _term(value: object, limit: int = MAX_TERM_CHARS) -> str | None:
    return clip(value, limit) if isinstance(value, str) and value.strip() else None


def parse_consolidation(raw: str, allowed_ids: set[str]) -> Consolidation:
    """Valida a saída do modelo; descarta entradas inválidas em vez de falhar o lote todo."""
    data = extract_json_object(raw)
    if not any(k in data for k in ("events", "knowledge", "obsolete")):
        raise ValueError("a resposta do consolidador não tem events/knowledge/obsolete")

    events: list[str] = []
    for item in data.get("events") or []:
        text = _term(item.get("text") if isinstance(item, dict) else item, MAX_EVENT_CHARS)
        if text and text not in events:
            events.append(text)

    facts: list[FactSpec] = []
    for item in data.get("knowledge") or []:
        if not isinstance(item, dict):
            continue
        subject, relation, obj = (_term(item.get(k)) for k in ("subject", "relation", "object"))
        if subject and relation and obj:
            spec = FactSpec(subject, relation, obj, _term(item.get("subject_type"), 40),
                            _term(item.get("object_type"), 40))
            if spec not in facts:
                facts.append(spec)

    obsolete = [i for i in (data.get("obsolete") or [])
                if isinstance(i, str) and i.strip().upper() in allowed_ids]
    return Consolidation(events[:MAX_EVENTS], facts[:MAX_FACTS],
                         list(dict.fromkeys(i.strip().upper() for i in obsolete)))


class Consolidator:
    def __init__(self, engine: ConsolidatorEngine, template: str, events: EventsMemory,
                 knowledge: KnowledgeMemory, persona_name: Callable[[], str],
                 similar_k: int = 5) -> None:
        self.engine, self.template = engine, template
        self.events, self.knowledge = events, knowledge
        self.persona_name, self.similar_k = persona_name, similar_k
        self.log = get_logger("consolidator", "core")
        self.prompt_log = get_prompt_logger("consolidator", "core")

    async def consolidate(self, messages: Sequence[Message]) -> ConsolidationResult:
        name = self.persona_name()
        conversation = "\n".join(
            f"[{m.created_at[:10]}] {'usuário' if m.role == 'user' else name}: "
            f"{clip(m.content, MAX_MESSAGE_CHARS)}" for m in messages)
        query = " ".join(m.content for m in messages)[-QUERY_CHARS:]

        # memórias parecidas: events e knowledge consultados em paralelo, sem marcar acesso
        hits, facts = await asyncio.gather(
            asyncio.to_thread(self.events.index.search, query, self.similar_k, False),
            asyncio.to_thread(self.knowledge.graph.search, query, self.similar_k, 2, False))
        allowed = {f"E{h.id}" for h in hits} | {f"F{f.id}" for f in facts}
        prompt = render(self.template, {
            "name": name, "messages": conversation,
            "events": "\n".join(f"[E{h.id}] {self.events.format(h)}" for h in hits) or "(nenhum)",
            "knowledge": "\n".join(f"[F{f.id}] {self.knowledge.format(f)}" for f in facts) or "(nenhum)"})
        self.prompt_log.info("%s", prompt)

        raw = await self.engine.run([{"role": "system", "content": prompt},
                                     {"role": "user", "content": "Consolide."}])
        result = await self._apply(parse_consolidation(raw, allowed), messages[-1].created_at[:10])
        self.log.info("consolidado: %s", result)
        return result

    async def _apply(self, plan: Consolidation, date: str) -> ConsolidationResult:
        for text in plan.events:
            await asyncio.to_thread(self.events.add, text, date)  # add pode chamar o embedder (rede)
        added_fact_ids: set[int] = set()
        for spec in plan.facts:
            fact = await asyncio.to_thread(self.knowledge.add_fact, spec.subject, spec.relation,
                                           spec.object, spec.subject_type, spec.object_type)
            added_fact_ids.add(fact.id)
        removed = 0
        for ident in plan.obsolete:
            kind, number = _ID.match(ident).groups()  # type: ignore[union-attr]
            number = int(number)
            if kind == "F" and number in added_fact_ids:
                continue  # o fato foi reforçado neste mesmo lote
            deleted = (self.events.delete(number) if kind == "E" else self.knowledge.delete(number))
            removed += int(deleted)
        return ConsolidationResult(len(plan.events), len(plan.facts), removed)
