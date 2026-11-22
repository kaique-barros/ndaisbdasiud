# Arquivos da persona em .data/personas/<uuid>/: descrições, state.md, scene.json, summary.md.
# Lidos a cada turno (edições valem na hora) e escritos de forma atômica, porque o summary em
# background escreve enquanto o próximo turno pode estar lendo.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DESCRIPTIONS = (("Personalidade", "personality.md"), ("Traços", "traits.md"),
                ("Aparência", "appearance.md"))


class PersonaFiles:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)

    def text(self, relative: str) -> str:
        path = self.dir / relative
        return path.read_text(encoding="utf-8").strip() if path.is_file() else ""

    def write_text(self, relative: str, content: str) -> None:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content.rstrip() + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def persona(self) -> str:
        sections = []
        for title, filename in DESCRIPTIONS:
            text = self.text(f"descriptions/{filename}")
            if text:
                sections.append(f"## {title}\n{text}")
        return "\n\n".join(sections)

    def scene_data(self) -> dict[str, Any]:
        raw = self.text("scene.json")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def scene(self) -> str:
        """Cenário em texto para o prompt (JSON inválido ou não-objeto vai como está)."""
        raw = self.text("scene.json")
        if not raw:
            return ""
        data = self.scene_data()
        if data:
            return "\n".join(f"- {k}: {v}" for k, v in data.items() if v not in ("", None))
        return raw if raw not in ("{}", "[]") else ""

    def write_scene(self, data: dict[str, Any]) -> None:
        self.write_text("scene.json", json.dumps(data, ensure_ascii=False, indent=2))

    @property
    def summary(self) -> str:
        return self.text("summary.md")

    @property
    def state(self) -> str:
        return self.text("state.md")
