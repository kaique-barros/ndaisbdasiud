# Consolidador de conhecimento: chamada ao modelo; o prompt, a validação e a aplicação
# ficam em core/consolidator.py (mesma relação de core/actor com engine/main/actor).
#
# Diferença para o summary: o summary só mantém o fio da conversa (summary.md/state.md/scene.json);
# aqui o que foi dito vira memória de longo prazo:
#   - events:    o que ACONTECEU, factual, sem impressões da persona
#   - knowledge: o que a persona APRENDEU (entidades e relações para o graph-rag),
#                com timestamp de criação e de último acesso
# As impressões pessoais continuam no notetaker (notes), não aqui.
#
# Entrada : lote de mensagens ainda não consolidadas + memórias parecidas já existentes (com ids),
#           para atualizar/mesclar em vez de duplicar
# Saída   : JSON com `events`, `knowledge` (novos) e `obsolete` (ids substituídos)
# Execução: em background (core/pipeline), idempotente via marcador `consolidated_up_to`.
from __future__ import annotations

from typing import Any

from src.engine.model import Messages, Model
from src.utils.logger import get_logger


class ConsolidatorEngine:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.log = get_logger("consolidator_engine", "engine")

    async def run(self, messages: Messages, **overrides: Any) -> str:
        response = await self.model.generate(messages, **overrides)
        text = response.text.strip()
        if not text:
            self.log.error("resposta vazia (finish=%s)", response.finish_reason)
            raise ValueError("o modelo devolveu uma resposta vazia")
        return text
