# Relação com engine/main/primary_interpreter igual à de core/actor com engine/main/actor:
# aqui se monta o prompt e se interpreta a resposta; o engine só chama o modelo. Os logs
# seguem a mesma lógica (prompt em interpreter.prompt.log, eventos em interpreter.log).
#
# O interpretador NÃO responde ao usuário. Devolve uma `Interpretation`:
#   - queries:       consultas por memória (events/knowledge/notes), feitas depois em paralelo
#   - tone:          tom sugerido para a resposta do actor
#   - emotion_delta: como a mensagem mexe com as emoções da persona
# Qualquer falha (modelo fora do ar, JSON inválido) cai num fallback seguro: a conversa
# segue, só sem o refinamento do interpretador.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from src.core.actor import render
from src.engine.main.primary_interpreter import PrimaryInterpreter
from src.memory.flash import FlashMemory
from src.utils.jsonx import extract_json_object
from src.utils.logger import get_logger, get_prompt_logger

MAX_QUERIES = 3
MAX_QUERY_CHARS = 300
MAX_TONE_CHARS = 200

STORE_HELP = {
    "events": "o que aconteceu: fatos de conversas e acontecimentos passados",
    "knowledge": "o que {{name}} aprendeu: pessoas, lugares, coisas e como se relacionam",
    "notes": "impressões e opiniões pessoais de {{name}} sobre pessoas, coisas e eventos",
}



@dataclass(frozen=True)
class Interpretation:
    queries: dict[str, list[str]]
    tone: str = ""
    emotion_delta: dict[str, float] = field(default_factory=dict)
    fallback: bool = False


@dataclass(frozen=True)
class InterpreterContext:
    persona_name: str
    user_messages: Sequence[str]
    persona: str = ""
    emotions: str = ""
    state: str = ""
    scene: str = ""
    summary: str = ""


def fallback_interpretation(user_messages: Sequence[str], store_names: Sequence[str]) -> Interpretation:
    """Sem interpretador: todas as memórias são consultadas com o texto do usuário."""
    text = "\n".join(m.strip() for m in user_messages if m.strip())[:MAX_QUERY_CHARS]
    return Interpretation({name: [text] for name in store_names} if text else {}, fallback=True)


def parse_interpretation(raw: str, store_names: Sequence[str],
                         user_messages: Sequence[str] = ()) -> Interpretation:
    """Valida e normaliza a saída do modelo. Lança ValueError se não houver JSON utilizável."""
    data = extract_json_object(raw)

    queries: dict[str, list[str]] = {}
    raw_queries = data.get("queries")
    if isinstance(raw_queries, dict):
        for name in store_names:
            value = raw_queries.get(name, [])
            items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
            cleaned: list[str] = []
            for q in items:
                q = " ".join(q.split())[:MAX_QUERY_CHARS] if isinstance(q, str) else ""
                if q and q not in cleaned:
                    cleaned.append(q)
            if cleaned:
                queries[name] = cleaned[:MAX_QUERIES]
    else:  # o modelo não deu consultas: o texto do usuário é o melhor palpite
        queries = fallback_interpretation(user_messages, store_names).queries

    tone = data.get("tone")
    tone = " ".join(tone.split())[:MAX_TONE_CHARS] if isinstance(tone, str) else ""

    delta: dict[str, float] = {}
    raw_delta = data.get("emotion_delta")
    if isinstance(raw_delta, dict):
        for key, value in raw_delta.items():
            if isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
                delta[key.strip().lower()] = float(value)
    return Interpretation(queries, tone, delta)


class InterpreterPromptBuilder:
    def __init__(self, template: str, flash: FlashMemory, store_names: Sequence[str],
                 emotion_labels: Mapping[str, str], max_chars: int = 12000, flash_k: int = 6) -> None:
        self.template, self.flash = template, flash
        self.store_names = list(store_names)
        self.emotion_labels = dict(emotion_labels)
        self.max_chars, self.flash_k = max_chars, flash_k
        self.log = get_logger("interpreter", "core")
        self.prompt_log = get_prompt_logger("interpreter", "core")

    def build(self, ctx: InterpreterContext) -> list[dict[str, str]]:
        user_text = "\n".join(m.strip() for m in ctx.user_messages if m.strip())
        if not user_text:
            raise ValueError("o interpretador precisa de ao menos uma mensagem do usuário")
        stores = "\n".join(f"- {n}: {STORE_HELP.get(n, 'memória da persona')}" for n in self.store_names)
        stores = render(stores, {"name": ctx.persona_name}) if stores else "(nenhuma)"
        values = {
            "name": ctx.persona_name, "persona": ctx.persona, "emotions": ctx.emotions,
            "state": ctx.state, "scene": ctx.scene, "summary": ctx.summary, "stores": stores,
            "emotion_keys": ", ".join(f"{k} ({v})" for k, v in self.emotion_labels.items()),
        }
        fixed = render(self.template, {**values, "flash": ""})
        budget = max(0, self.max_chars - len(fixed) - len(user_text))
        flash_text = self.flash.format(k=self.flash_k, max_chars=budget)
        system = render(self.template, {**values, "flash": flash_text})
        self.log.info("prompt montado: %d chars", len(system) + len(user_text))
        self.prompt_log.info("=== SYSTEM ===\n%s\n=== USER ===\n%s", system, user_text)
        return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]


class Interpreter:
    """Prompt (core) + chamada ao modelo (engine) + interpretação com fallback. Nunca lança."""

    def __init__(self, builder: InterpreterPromptBuilder, engine: PrimaryInterpreter,
                 store_names: Sequence[str], enabled: bool = True) -> None:
        self.builder, self.engine = builder, engine
        self.store_names, self.enabled = list(store_names), enabled
        self.log = get_logger("interpreter", "core")

    async def interpret(self, ctx: InterpreterContext) -> Interpretation:
        if not self.enabled:
            return fallback_interpretation(ctx.user_messages, self.store_names)
        try:
            raw = await self.engine.interpret(self.builder.build(ctx))
            result = parse_interpretation(raw, self.store_names, ctx.user_messages)
        except Exception as exc:
            self.log.warning("interpretador falhou (%s: %s); usando o fallback",
                             type(exc).__name__, exc)
            return fallback_interpretation(ctx.user_messages, self.store_names)
        self.log.info("consultas=%s tom=%r delta=%s", {k: len(v) for k, v in result.queries.items()},
                      result.tone, result.emotion_delta)
        return result
