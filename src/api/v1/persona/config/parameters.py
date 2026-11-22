# Parâmetros da persona (parameters.json). O nome fica sincronizado com o registro (menu).
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from src.api.deps import get_state
from src.api.state import AppState
from src.utils.config import load_json

router = APIRouter(prefix="/v1/personas/{persona_id}/config/parameters", tags=["config"])


@router.get("")
def get_parameters(persona_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    state.record(persona_id)
    return load_json(state.config.paths.persona_dir(persona_id) / "parameters.json")


@router.put("")
def put_parameters(persona_id: str, body: dict[str, Any],
                   state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Substitui todos os parâmetros. Sem 'name', mantém o nome atual."""
    return state.write_parameters(persona_id, body)
