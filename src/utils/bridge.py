# Adaptador de rede para as APIs de IA. A maioria recebe o mesmo formato de dado
# (chat completions no estilo OpenAI), então o resto do sistema fala só com este módulo.
# Só biblioteca padrão: o transporte HTTP é injetável (facilita testes e trocar por httpx depois).
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.utils.logger import get_logger

# (url, headers, body_bytes, timeout_s) -> (status, body_bytes)
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
USER_AGENT = "persona-engine/0.1 (+https://github.com/kaique-barros/ndaisbdasiud)"


class BridgeError(Exception):
    """Falha na chamada ao provedor. `status` é o HTTP, quando houver."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes,
                     timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BridgeError(f"falha de rede: {exc}", retryable=True) from exc


class OpenAICompatibleBridge:
    """Cliente de /chat/completions para qualquer provedor compatível com a API da OpenAI."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 60.0,
                 max_retries: int = 2, backoff_s: float = 0.5,
                 transport: Transport = urllib_transport) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.transport = transport
        self.log = get_logger("bridge", "utils")

    @classmethod
    def from_provider_config(cls, cfg: Mapping[str, Any],
                             transport: Transport = urllib_transport) -> "OpenAICompatibleBridge":
        """Cria a partir de .configs/providers/<nome>.json (a chave vem da variável de ambiente)."""
        if "base_url" not in cfg:
            raise BridgeError("config do provider sem base_url")
        env = cfg.get("api_key_env")
        return cls(
            base_url=cfg["base_url"],
            api_key=os.environ.get(env) if env else None,
            timeout_s=float(cfg.get("timeout_s", 60)),
            max_retries=int(cfg.get("max_retries", 2)),
            transport=transport,
        )

    def chat_sync(self, model: str, messages: Sequence[Mapping[str, str]],
                  **params: Any) -> ChatResponse:
        payload = {"model": model, "messages": list(messages),
                   **{k: v for k, v in params.items() if v is not None}}
        return self._parse(self._post("/chat/completions", payload))

    def embed_sync(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        """Embeddings (/embeddings, formato OpenAI). Na ordem dos textos."""
        raw = self._post("/embeddings", {"model": model, "input": list(texts)})
        try:
            items = sorted(json.loads(raw)["data"], key=lambda d: d["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (ValueError, KeyError, TypeError) as exc:
            raise BridgeError(f"resposta de embeddings inesperada: {exc}") from exc
        if len(vectors) != len(texts):
            raise BridgeError("o provedor devolveu um número diferente de embeddings")
        return vectors

    def _post(self, path: str, payload: Mapping[str, Any]) -> bytes:
        """POST JSON com retry/backoff; devolve o corpo da resposta 200."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # User-Agent próprio: o padrão do urllib ("Python-urllib/3.x") é bloqueado pelo Cloudflare
        # que fica na frente de provedores como a Groq (HTTP 403, "error code: 1010").
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}{path}"

        last: BridgeError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                status, data = self.transport(url, headers, body, self.timeout_s)
            except BridgeError as exc:
                last = exc
            else:
                if status == 200:
                    return data
                last = BridgeError(f"HTTP {status}: {data[:300].decode('utf-8', 'replace')}",
                                   status=status, retryable=status in RETRYABLE_STATUS)
            if not last.retryable or attempt == self.max_retries:
                break
            delay = self.backoff_s * (2 ** attempt)
            self.log.warning("tentativa %d falhou (%s); nova em %.2fs", attempt + 1, last, delay)
            time.sleep(delay)
        assert last is not None
        self.log.error("chamada falhou: %s", last)
        raise last

    async def chat(self, model: str, messages: Sequence[Mapping[str, str]],
                   **params: Any) -> ChatResponse:
        """Versão assíncrona: roda a chamada bloqueante em thread, sem travar o loop."""
        return await asyncio.to_thread(self.chat_sync, model, messages, **params)

    @staticmethod
    def _parse(data: bytes) -> ChatResponse:
        try:
            obj = json.loads(data)
            choice = obj["choices"][0]
            text = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BridgeError(f"resposta inesperada do provedor: {exc}") from exc
        usage = obj.get("usage") or {}
        return ChatResponse(
            text=text,
            model=obj.get("model"),
            finish_reason=choice.get("finish_reason"),
            usage={k: v for k, v in usage.items() if isinstance(v, int)},
            raw=obj,
        )
