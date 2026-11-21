# O ator principal: envia o prompt montado por core/actor ao modelo e devolve a fala da persona.
# Este log (logs/engine/actor.log) fica só com conexão, tempo e erros; o conteúdo do prompt
# é logado em core/actor.
from __future__ import annotations

from typing import Any

from src.engine.model import Messages, Model
from src.utils.logger import get_logger


class ActorError(Exception):
    """O modelo respondeu, mas sem uma fala utilizável."""


class Actor:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.log = get_logger("actor_main", "engine")

    async def respond(self, messages: Messages, **overrides: Any) -> str:
        response = await self.model.generate(messages, **overrides)
        text = response.text.strip()
        if not text:
            self.log.error("resposta vazia (finish=%s)", response.finish_reason)
            raise ActorError("o modelo devolveu uma resposta vazia")
        return text
