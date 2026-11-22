# Modelos de entrada/saída da API (validação e limites de tamanho).
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_CHARS = 8000


class PersonaOut(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    last_conversation_at: str | None = None
    metadata: dict[str, Any] = {}


class PersonaCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("o nome não pode ser vazio")
        return v


class PersonaPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    status: Literal["active", "archived"] | None = None


class ChatIn(BaseModel):
    messages: list[str] = Field(min_length=1, max_length=20)
    conversation_id: str | None = Field(None, max_length=100)

    @field_validator("messages")
    @classmethod
    def _check(cls, v: list[str]) -> list[str]:
        if any(len(m) > MAX_MESSAGE_CHARS for m in v):
            raise ValueError(f"cada mensagem pode ter no máximo {MAX_MESSAGE_CHARS} caracteres")
        if not any(m.strip() for m in v):
            raise ValueError("envie ao menos uma mensagem não vazia")
        return v


class ChatOut(BaseModel):
    reply: str
    persona: str


class DescriptionsIn(BaseModel):
    """Campos omitidos (null) ficam como estão; string vazia limpa o campo."""
    personality: str | None = Field(None, max_length=20000)
    traits: str | None = Field(None, max_length=20000)
    appearance: str | None = Field(None, max_length=20000)


class EventIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    when: str | None = Field(None, max_length=40)


class FactIn(BaseModel):
    subject: str = Field(min_length=1, max_length=100)
    relation: str = Field(min_length=1, max_length=100)
    object: str = Field(min_length=1, max_length=100)
    subject_type: str | None = Field(None, max_length=40)
    object_type: str | None = Field(None, max_length=40)


class NoteIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    subject: str | None = Field(None, max_length=100)
    importance: float | None = Field(None, ge=0, le=1)
