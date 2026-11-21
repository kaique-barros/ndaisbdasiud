# Modelo abstrato: enviar / receber chamadas à API, com configs padrão vindas de
# .configs/engine/<componente>.json (+ provider). Cada engine (actor, notetaker...) usa um Model.
# O conteúdo dos prompts é logado pelas classes de core/; aqui só vão eventos de conexão e erros.
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from src.utils.bridge import ChatResponse, OpenAICompatibleBridge
from src.utils.logger import get_logger

Messages = Sequence[Mapping[str, str]]

# parâmetros de geração que podem vir da config e ser sobrescritos por chamada
GENERATION_KEYS = ("temperature", "max_tokens", "top_p", "stop", "seed")


class Model(ABC):
    def __init__(self, name: str, defaults: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self.defaults = {k: v for k, v in (defaults or {}).items() if k in GENERATION_KEYS}
        self.log = get_logger(name, "engine")

    @abstractmethod
    async def _complete(self, messages: Messages, params: dict[str, Any]) -> ChatResponse:
        """Faz a chamada real ao provedor."""

    async def generate(self, messages: Messages, **overrides: Any) -> ChatResponse:
        params = {**self.defaults, **{k: v for k, v in overrides.items() if k in GENERATION_KEYS}}
        t0 = time.perf_counter()
        try:
            response = await self._complete(messages, params)
        except Exception as exc:
            self.log.error("falha após %.2fs: %s", time.perf_counter() - t0, exc)
            raise
        self.log.info("ok em %.2fs (finish=%s, usage=%s)", time.perf_counter() - t0,
                      response.finish_reason, response.usage)
        return response

    async def ask(self, user: str, system: str | None = None, **overrides: Any) -> str:
        """Atalho: um system opcional + uma mensagem do usuário -> texto da resposta."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return (await self.generate(messages, **overrides)).text


class ChatModel(Model):
    """Model que fala com um provedor compatível com a API da OpenAI via bridge."""

    def __init__(self, name: str, model_id: str, bridge: OpenAICompatibleBridge,
                 defaults: Mapping[str, Any] | None = None) -> None:
        super().__init__(name, defaults)
        self.model_id = model_id
        self.bridge = bridge

    @classmethod
    def from_configs(cls, name: str, engine_cfg: Mapping[str, Any],
                     provider_cfg: Mapping[str, Any],
                     bridge: OpenAICompatibleBridge | None = None) -> "ChatModel":
        """engine_cfg: .configs/engine/<name>.json; provider_cfg: .configs/providers/<provider>.json."""
        if "model" not in engine_cfg:
            raise ValueError(f"config de engine '{name}' sem o campo 'model'")
        bridge = bridge or OpenAICompatibleBridge.from_provider_config(provider_cfg)
        return cls(name, engine_cfg["model"], bridge, engine_cfg)

    async def _complete(self, messages: Messages, params: dict[str, Any]) -> ChatResponse:
        return await self.bridge.chat(self.model_id, messages, **params)
