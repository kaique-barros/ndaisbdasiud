# Recebe as infos relevantes (memória, emoção, cenário...) e monta o prompt geral do actor.
# Por que existe separado do engine/main/actor:
#   1. o conteúdo do prompt é logado aqui (logs/core/actor.prompt.log), deixando o log do
#      engine só com conexão e erros; este arquivo tem também um log resumido de eventos;
#   2. o tamanho da memória flash é ajustado ao tamanho dos outros inputs (orçamento em caracteres).
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from src.memory.flash import FlashMemory
from src.utils.logger import get_logger, get_prompt_logger

EMPTY = "(nenhum)"
_TOKEN = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class ActorContext:
    persona_name: str
    user_messages: Sequence[str]
    persona: str = ""    # descrições (personalidade, traços, aparência)
    emotions: str = ""   # estado emocional atual, já descrito em texto
    tone: str = ""       # tom sugerido pelo interpretador para esta resposta
    state: str = ""      # state.md
    scene: str = ""      # scene.json já renderizado
    summary: str = ""    # resumo da conversa atual
    memory: str = ""     # saída de core/memories (events, knowledge, notes)


def render(template: str, values: dict[str, str]) -> str:
    """Troca {{chave}} em uma única passada (um valor que contenha {{x}} não é reexpandido).
    Valores vazios viram EMPTY; chaves desconhecidas ficam como estão."""
    return _TOKEN.sub(lambda m: (values[m.group(1)].strip() or EMPTY) if m.group(1) in values
                      else m.group(0), template)


class ActorPromptBuilder:
    def __init__(self, template: str, flash: FlashMemory, max_chars: int = 24000) -> None:
        self.template = template
        self.flash = flash
        self.max_chars = max_chars
        self.log = get_logger("actor", "core")
        self.prompt_log = get_prompt_logger("actor", "core")

    def build(self, ctx: ActorContext) -> list[dict[str, str]]:
        """Devolve as mensagens (system + user) para o modelo. Deve ser chamado ANTES de
        gravar as mensagens do turno atual na flash, para a flash trazer só o passado."""
        values = {
            "name": ctx.persona_name, "persona": ctx.persona, "emotions": ctx.emotions,
            "tone": ctx.tone, "state": ctx.state, "scene": ctx.scene, "summary": ctx.summary,
            "memory": ctx.memory,
        }
        user_text = "\n".join(m.strip() for m in ctx.user_messages if m.strip())
        if not user_text:
            raise ValueError("o actor precisa de ao menos uma mensagem do usuário")

        # a flash recebe o que sobra do orçamento depois dos demais inputs
        fixed = render(self.template, {**values, "flash": ""})
        budget = max(0, self.max_chars - len(fixed) - len(user_text))
        flash_text = self.flash.format(max_chars=budget)
        system = render(self.template, {**values, "flash": flash_text})

        self.log.info("prompt montado: %d chars (flash %d de %d disponíveis)",
                      len(system) + len(user_text), len(flash_text), budget)
        self.prompt_log.info("=== SYSTEM ===\n%s\n=== USER ===\n%s", system, user_text)
        return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
