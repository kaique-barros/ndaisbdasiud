# Extração tolerante de JSON da saída de um modelo (bloco de código, texto ao redor).
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json_object(raw: str) -> dict:
    """Devolve o objeto JSON da resposta; lança ValueError se não houver um utilizável."""
    text = _FENCE.sub("", raw.strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("a resposta do modelo não contém JSON")
        data = json.loads(text[start:end + 1])  # JSON cercado de texto
    if not isinstance(data, dict):
        raise ValueError("a resposta do modelo deve ser um objeto JSON")
    return data


def clip(text: str, limit: int) -> str:
    """Corta em `limit` caracteres, preferindo o fim de uma palavra."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut.rstrip(",;:. ") + "…"
