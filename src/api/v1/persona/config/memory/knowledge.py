# Conhecimento (graph-RAG) da persona: fatos como relações entre entidades.
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_state
from src.api.schemas import FactIn
from src.api.state import AppState

router = APIRouter(prefix="/v1/personas/{persona_id}/config/memory/knowledge", tags=["memory"])


def _store(state: AppState, persona_id: str):
    store = state.orchestrator(persona_id).memory.stores.get("knowledge")
    if store is None:
        raise HTTPException(404, "memória 'knowledge' indisponível")
    return store


@router.get("")
def list_facts(persona_id: str, q: str | None = Query(None, max_length=300),
               limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
               state: AppState = Depends(get_state)) -> list[dict]:
    """Com `q`, acha entidades pelo nome e expande a vizinhança (sem contar como acesso)."""
    store = _store(state, persona_id)
    facts = store.graph.search(q, limit, touch=False) if q else store.list(limit, offset)
    return [asdict(f) for f in facts]


@router.post("", status_code=201)
def create_fact(persona_id: str, body: FactIn, state: AppState = Depends(get_state)) -> dict:
    return asdict(_store(state, persona_id).add_fact(
        body.subject, body.relation, body.object, body.subject_type, body.object_type))


@router.delete("/{fact_id}", status_code=204)
def delete_fact(persona_id: str, fact_id: int, state: AppState = Depends(get_state)) -> None:
    if not _store(state, persona_id).delete(fact_id):
        raise HTTPException(404, "fato não encontrado")
