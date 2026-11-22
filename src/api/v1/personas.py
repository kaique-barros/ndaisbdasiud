# Personas: registro (menu), conversa, fim de conversa e estado atual.
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from src.api.deps import get_state
from src.api.schemas import ChatIn, ChatOut, PersonaCreate, PersonaOut, PersonaPatch
from src.api.state import AppState
from src.data.personas import ARCHIVED, create_persona
from src.utils.config import load_json

router = APIRouter(prefix="/v1/personas", tags=["personas"])


@router.get("", response_model=list[PersonaOut])
def list_personas(status: Literal["active", "archived"] | None = None,
                  search: str | None = Query(None, max_length=100),
                  state: AppState = Depends(get_state)):
    """Mais recentemente conversadas primeiro (é o que o menu precisa; sem abrir cada persona)."""
    return [asdict(p) for p in state.registry.list(status, search)]


@router.post("", response_model=PersonaOut, status_code=201)
def create(body: PersonaCreate, state: AppState = Depends(get_state)):
    return asdict(create_persona(state.registry, state.config.paths, body.name))


@router.get("/{persona_id}", response_model=PersonaOut)
def get_persona(persona_id: str, state: AppState = Depends(get_state)):
    return asdict(state.record(persona_id))


@router.patch("/{persona_id}", response_model=PersonaOut)
def patch_persona(persona_id: str, body: PersonaPatch, state: AppState = Depends(get_state)):
    state.record(persona_id)
    if body.name is not None:
        current = load_json(state.config.paths.persona_dir(persona_id) / "parameters.json")
        state.write_parameters(persona_id, {**current, "name": body.name})
    if body.status is not None:
        state.registry.set_status(persona_id, body.status)
    return asdict(state.record(persona_id))


@router.post("/{persona_id}/chat", response_model=ChatOut)
async def chat(persona_id: str, body: ChatIn, state: AppState = Depends(get_state)):
    record = await run_in_threadpool(state.record, persona_id)
    if record.status == ARCHIVED:
        raise HTTPException(409, "persona arquivada: reative-a para conversar")
    orch = await run_in_threadpool(state.orchestrator, persona_id)
    reply = await orch.chat(body.messages, body.conversation_id)
    return ChatOut(reply=reply, persona=orch.name)


@router.post("/{persona_id}/finalize", status_code=204)
async def finalize(persona_id: str, state: AppState = Depends(get_state)):
    """Fim da conversa: processa agora o que o background ainda não processou."""
    orch = await run_in_threadpool(state.orchestrator, persona_id)
    if orch.pipeline:
        await orch.pipeline.flush()


@router.get("/{persona_id}/status")
def status(persona_id: str, state: AppState = Depends(get_state)):
    """Emoções atuais (com decaimento aplicado), resumo, estado e cenário."""
    orch = state.orchestrator(persona_id)
    emotions = orch.emotion_store.load()  # estado novo, sem mexer no que está em uso por um turno
    return {
        "emotions": {
            "values": {k: round(v, 3) for k, v in emotions.values.items()},
            "active": [{"emotion": e, "label": emotions.level_name(e, v), "value": round(v, 3)}
                       for e, v in emotions.active()],
            "dyads": [asdict(d) for d in emotions.dyads()],
        },
        "summary": orch.files.summary,
        "state": orch.files.state,
        "scene": orch.files.scene_data(),
    }
