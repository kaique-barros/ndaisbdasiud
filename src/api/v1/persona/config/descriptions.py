# Descrições da persona (personalidade, traços, aparência), relidas a cada turno.
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_state
from src.api.schemas import DescriptionsIn
from src.api.state import AppState

router = APIRouter(prefix="/v1/personas/{persona_id}/config/descriptions", tags=["config"])

FIELDS = {"personality": "personality.md", "traits": "traits.md", "appearance": "appearance.md"}


def _read(state: AppState, persona_id: str) -> dict[str, str]:
    files = state.files(persona_id)
    return {field: files.text(f"descriptions/{name}") for field, name in FIELDS.items()}


@router.get("")
def get_descriptions(persona_id: str, state: AppState = Depends(get_state)) -> dict[str, str]:
    return _read(state, persona_id)


@router.put("")
def put_descriptions(persona_id: str, body: DescriptionsIn,
                     state: AppState = Depends(get_state)) -> dict[str, str]:
    files = state.files(persona_id)
    for field, name in FIELDS.items():
        value = getattr(body, field)
        if value is not None:
            files.write_text(f"descriptions/{name}", value)
    return _read(state, persona_id)
