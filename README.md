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
| `.configs/providers/` | um JSON por provedor de LLM, todos no formato OpenAI-compatível (`ollama.json` local, `groq.json` nuvem) |
| `.data/global.db` | registro de personas (uuid, nome, status, última conversa, metadados) |
| `.data/personas/<uuid>/` | descrições, `parameters.json`, `emotions.json`, `scene.json`, `state.md` e `memories/` (bancos criados em execução) |
| `.logs/` | `general.log` + `<categoria>/<classe>.log` + `<classe>.prompt.log` (prompts ficam fora do log geral) |

## Modelos: local (Ollama) ou nuvem (Groq)

Todos os componentes usam o mesmo provider, escolhido pela variável `PERSONA_PROVIDER`
(padrão: `ollama`). Copie `.env.example` para `.env` ou exporte as variáveis.

| | Local | Nuvem |
|---|---|---|
| `PERSONA_PROVIDER` | `ollama` | `groq` |
| Pré-requisito | `ollama serve` e `ollama pull llama3.2` | `GROQ_API_KEY` ([console.groq.com/keys](https://console.groq.com/keys)) |
| Modelo padrão | `llama3.2` | `llama-3.3-70b-versatile` |

O modelo padrão fica em `default_model` no JSON do provider. Para usar outro modelo em um componente,
adicione ao `.configs/engine/<componente>.json` um `"models": {"groq": "...", "ollama": "..."}`
(ou `"model": "..."` para valer em qualquer provider).

## Conversando (CLI)

```
python -m src.cli --new "Ana"     # cria a persona a partir de .data/personas/persona_example
python -m src.cli --list          # lista as personas
python -m src.cli <uuid>          # conversa (/sair encerra)
```

Edite as descrições em `.data/personas/<uuid>/descriptions/`; elas são relidas a cada turno.
Hoje o turno usa: descrições, `state.md`, `scene.json`, memória flash e as memórias pesquisáveis que
forem registradas. Ainda não ligados: `primary_interpreter` (por enquanto a consulta de memória é o
próprio texto do usuário), emoções, `notetaker`, `summary` e `consolidator`.

## Desenvolvimento

```
python -m pytest        # ou: python -m unittest discover -s tests
```

Python >= 3.11, sem dependências de runtime por enquanto.
