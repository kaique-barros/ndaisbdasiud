# Eventos (RAG factual) da persona: listar/buscar, criar e apagar.
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_state
from src.api.schemas import EventIn
from src.api.state import AppState

router = APIRouter(prefix="/v1/personas/{persona_id}/config/memory/events", tags=["memory"])


def _store(state: AppState, persona_id: str):
    store = state.orchestrator(persona_id).memory.stores.get("events")
    if store is None:
        raise HTTPException(404, "memória 'events' indisponível")
    return store


def _out(hit) -> dict:
    return {"id": hit.id, "text": hit.text, "when": hit.meta.get("when"),
            "created_at": hit.created_at, "last_accessed_at": hit.last_accessed_at}


@router.get("")
def list_events(persona_id: str, q: str | None = Query(None, max_length=300),
                limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                state: AppState = Depends(get_state)) -> list[dict]:
    """Com `q`, busca por relevância (sem contar como acesso); sem `q`, do mais novo ao mais antigo."""
    store = _store(state, persona_id)
    hits = store.index.search(q, limit, touch=False) if q else store.list(limit, offset)
    return [_out(h) for h in hits]


@router.post("", status_code=201)
def create_event(persona_id: str, body: EventIn, state: AppState = Depends(get_state)) -> dict:
    return {"id": _store(state, persona_id).add(body.text, body.when)}


@router.delete("/{event_id}", status_code=204)
def delete_event(persona_id: str, event_id: int, state: AppState = Depends(get_state)) -> None:
    if not _store(state, persona_id).delete(event_id):
        raise HTTPException(404, "evento não encontrado")
