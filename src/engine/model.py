# Modelo abstrato: enviar / receber chamadas à API, com configs padrão vindas de
# .configs/engine/<componente>.json (+ provider). Cada engine (actor, notetaker...) usa um Model.
# O conteúdo dos prompts é logado pelas classes de core/; aqui só vão eventos de conexão e erros.
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from src.data.rag import Embedder
from src.utils.bridge import ChatResponse, OpenAICompatibleBridge
from src.utils.config import Config
from src.utils.logger import get_logger

DEFAULT_PROVIDER = "ollama"

Messages = Sequence[Mapping[str, str]]

# parâmetros de geração que podem vir da config e ser sobrescritos por chamada
GENERATION_KEYS = ("temperature", "max_tokens", "top_p", "stop", "seed")


def resolve_provider(config: Config, engine_cfg: Mapping[str, Any] | None = None
                     ) -> tuple[str, dict[str, Any]]:
    """Provider: variável PERSONA_PROVIDER > campo "provider" do engine > "ollama"."""
    provider = (os.environ.get("PERSONA_PROVIDER") or (engine_cfg or {}).get("provider")
                or DEFAULT_PROVIDER)
    return provider, {"name": provider, **config.provider(provider)}


def make_embedder(config: Config, model_id: str) -> Embedder:
    """Função de embeddings (endpoint /embeddings do provider atual) para os RAGs."""
    _, provider_cfg = resolve_provider(config)
    bridge = OpenAICompatibleBridge.from_provider_config(provider_cfg)
    return lambda texts: bridge.embed_sync(model_id, texts)


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
    def from_name(cls, name: str, config: Config,
                  bridge: OpenAICompatibleBridge | None = None) -> "ChatModel":
        """Monta o modelo de um componente (ex.: "actor") a partir das configs.
        Provider: variável PERSONA_PROVIDER > campo "provider" do engine > "ollama".
        Assim a mesma config roda local (ollama) ou na nuvem (groq) trocando só a variável."""
        engine_cfg = config.engine(name)
        _, provider_cfg = resolve_provider(config, engine_cfg)
        return cls.from_configs(name, engine_cfg, provider_cfg, bridge)

    @classmethod
    def from_configs(cls, name: str, engine_cfg: Mapping[str, Any],
                     provider_cfg: Mapping[str, Any],
                     bridge: OpenAICompatibleBridge | None = None) -> "ChatModel":
        """Modelo usado, por prioridade: engine.models[<provider>] > engine.model >
        provider.default_model."""
        provider = provider_cfg.get("name", "")
        model_id = ((engine_cfg.get("models") or {}).get(provider)
                    or engine_cfg.get("model") or provider_cfg.get("default_model"))
        if not model_id:
            raise ValueError(f"nenhum modelo definido para '{name}' (provider '{provider}'): "
                             "use 'models'/'model' no engine ou 'default_model' no provider")
        bridge = bridge or OpenAICompatibleBridge.from_provider_config(provider_cfg)
        return cls(name, model_id, bridge, engine_cfg)

    async def _complete(self, messages: Messages, params: dict[str, Any]) -> ChatResponse:
        return await self.bridge.chat(self.model_id, messages, **params)
