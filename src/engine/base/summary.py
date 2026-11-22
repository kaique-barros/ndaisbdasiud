# O resumidor da conversa. É realmente só um resumo para não perder o fio da meada; a
# consolidação do conhecimento é em outra classe (consolidator). Mantém três arquivos da persona:
#   - summary.md  resumo corrido da conversa (limitado a `max_chars`)
#   - state.md    como a persona está agora (humor, energia, o que ocupa a cabeça)
#   - scene.json  cenário atual (local, momento, clima, quem está presente...)
# Escrita atômica, e só depois de a resposta do modelo ser validada: saída ruim não apaga nada.
from __future__ import annotations

from typing import Callable, Sequence

from src.core.actor import render
from src.core.persona_files import PersonaFiles
from src.engine.model import Model
from src.memory.flash import Message
from src.utils.jsonx import clip, extract_json_object
from src.utils.logger import get_logger, get_prompt_logger

MAX_MESSAGE_CHARS = 1200
MAX_STATE_CHARS = 600


def format_messages(messages: Sequence[Message], persona_name: str) -> str:
    return "\n".join(
        f"{'usuário' if m.role == 'user' else persona_name}: {clip(m.content, MAX_MESSAGE_CHARS)}"
        for m in messages)


class Summarizer:
    def __init__(self, model: Model, template: str, files: PersonaFiles,
                 persona_name: Callable[[], str], emotions: Callable[[], str] = lambda: "",
                 max_chars: int = 2000) -> None:
        self.model, self.template, self.files = model, template, files
        self.persona_name, self.emotions, self.max_chars = persona_name, emotions, max_chars
        self.log = get_logger("summary", "engine")
        self.prompt_log = get_prompt_logger("summary", "engine")

    async def update(self, messages: Sequence[Message]) -> None:
        name = self.persona_name()
        scene = self.files.scene()
        prompt = render(self.template, {
            "name": name, "summary": self.files.summary, "state": self.files.state,
            "emotions": self.emotions(), "scene": scene,
            "messages": format_messages(messages, name), "max_chars": str(self.max_chars)})
        self.prompt_log.info("%s", prompt)
        response = await self.model.generate([{"role": "system", "content": prompt},
                                              {"role": "user", "content": "Atualize."}])

        data = extract_json_object(response.text)  # valida tudo antes de escrever qualquer arquivo
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("a resposta do resumo não tem 'summary'")
        state, new_scene = data.get("state"), data.get("scene")

        self.files.write_text("summary.md", clip(summary, self.max_chars))
        if isinstance(state, str) and state.strip():
            self.files.write_text("state.md", clip(state, MAX_STATE_CHARS))
        if isinstance(new_scene, dict) and new_scene:
            self.files.write_scene({str(k): v for k, v in new_scene.items()
                                    if isinstance(v, (str, int, float)) and not isinstance(v, bool)})
        self.log.info("resumo atualizado com %d mensagens (%d chars)", len(messages), len(summary))
