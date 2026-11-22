# Memória flash (mensagens recentes): consultar e apagar.
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_state
from src.api.state import AppState

router = APIRouter(prefix="/v1/personas/{persona_id}/config/memory/messages", tags=["memory"])


@router.get("")
def list_messages(persona_id: str, limit: int = Query(50, ge=1, le=500),
                  conversation_id: str | None = Query(None, max_length=100),
                  state: AppState = Depends(get_state)) -> list[dict]:
    """As últimas mensagens em ordem cronológica, com a variação emocional de cada uma."""
    flash = state.orchestrator(persona_id).flash
    return [asdict(m) for m in flash.last(limit, conversation_id)]


@router.delete("/{message_id}", status_code=204)
def delete_message(persona_id: str, message_id: int, state: AppState = Depends(get_state)) -> None:
    if not state.orchestrator(persona_id).flash.delete(message_id):
        raise HTTPException(404, "mensagem não encontrada")
