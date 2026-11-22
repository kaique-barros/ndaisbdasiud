# O interpretador primário: recebe o prompt montado por core/interpreter e devolve o texto bruto
# do modelo (um JSON, validado em core/interpreter). Este log fica só com conexão, tempo e erros.
from __future__ import annotations

from typing import Any

from src.engine.model import Messages, Model
from src.utils.logger import get_logger


class InterpreterError(Exception):
    """O modelo respondeu, mas sem conteúdo."""


class PrimaryInterpreter:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.log = get_logger("primary_interpreter", "engine")

    async def interpret(self, messages: Messages, **overrides: Any) -> str:
        response = await self.model.generate(messages, **overrides)
        text = response.text.strip()
        if not text:
            self.log.error("resposta vazia (finish=%s)", response.finish_reason)
            raise InterpreterError("o modelo devolveu uma resposta vazia")
        return text
