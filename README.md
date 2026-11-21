# Persona Engine

Motor de persona conversacional com memória e emoções.

## Fluxo de um turno

1. `engine/main/primary_interpreter` lê a mensagem e gera as consultas de memória + o tom.
2. `core/memories` dispara **todas as buscas em paralelo** (events, knowledge, notes) e monta o prompt de memória.
3. `core/actor` monta o prompt final (estado, cenário, memórias, flash, resumo, emoções) e `engine/main/actor` responde.
4. Em background: `notetaker` (notas), `summary` (`scene.json`/`state.md`), `consolidator` (events + knowledge) e decaimento das emoções.

## Organização

| Pasta | Conteúdo |
|---|---|
| `.configs/engine/` | um JSON por componente (`actor`, `notetaker`, `consolidator`, `flash_memory`, `events_memory`...) |
| `.configs/prompts/` | um `.md` por modelo (`actor`, `primary_interpreter`, `summary`, `notetaker`, `consolidator`) |
| `.configs/providers/` | um JSON por provedor de LLM (`example.json` é o modelo) |
| `.data/global.db` | registro de personas (uuid, nome, status, última conversa, metadados) |
| `.data/personas/<uuid>/` | descrições, `parameters.json`, `emotions.json`, `scene.json`, `state.md` e `memories/` (bancos criados em execução) |
| `.logs/` | `general.log` + `<categoria>/<classe>.log` + `<classe>.prompt.log` (prompts ficam fora do log geral) |

## Desenvolvimento

```
python -m pytest        # ou: python -m unittest discover -s tests
```

Python >= 3.11, sem dependências de runtime por enquanto.
