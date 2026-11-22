# Notas pessoais da persona (impressões): listar/buscar, criar e apagar.
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_state
from src.api.schemas import NoteIn
from src.api.state import AppState

router = APIRouter(prefix="/v1/personas/{persona_id}/config/memory/notes", tags=["memory"])


def _store(state: AppState, persona_id: str):
    store = state.orchestrator(persona_id).memory.stores.get("notes")
    if store is None:
        raise HTTPException(404, "memória 'notes' indisponível")
    return store


def _out(hit) -> dict:
    return {"id": hit.id, "text": hit.text, "subject": hit.meta.get("subject"),
            "importance": hit.meta.get("importance"), "emotions": hit.meta.get("emotions", {}),
            "created_at": hit.created_at, "last_accessed_at": hit.last_accessed_at}


@router.get("")
def list_notes(persona_id: str, q: str | None = Query(None, max_length=300),
               limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
               state: AppState = Depends(get_state)) -> list[dict]:
    store = _store(state, persona_id)
    hits = store.index.search(q, limit, touch=False) if q else store.list(limit, offset)
    return [_out(h) for h in hits]


@router.post("", status_code=201)
def create_note(persona_id: str, body: NoteIn, state: AppState = Depends(get_state)) -> dict:
    return {"id": _store(state, persona_id).add(body.text, body.subject, body.importance)}


@router.delete("/{note_id}", status_code=204)
def delete_note(persona_id: str, note_id: int, state: AppState = Depends(get_state)) -> None:
    if not _store(state, persona_id).delete(note_id):
        raise HTTPException(404, "nota não encontrada")
