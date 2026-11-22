# Persona Engine

Motor de persona conversacional com **memória** e **emoções**. Cada persona conversa, sente, lembra
e aprende com o tempo; tudo roda local (Ollama) ou na nuvem (Groq) e só usa a biblioteca padrão do
Python (a API HTTP é opcional).

## Como um turno funciona

1. **Interpretador** (`engine/main/primary_interpreter`) lê a mensagem, as emoções e o contexto e
   devolve em JSON: consultas de memória por tipo, o tom da resposta e a variação emocional.
   Se falhar, o turno segue com um fallback (a consulta vira o próprio texto do usuário).
2. As **emoções** são atualizadas e **todas as buscas de memória rodam em paralelo**
   (events, knowledge, notes). Uma memória lenta ou quebrada não derruba as outras.
3. **Actor** (`core/actor` monta o prompt, `engine/main/actor` chama o modelo) responde.
4. O turno vai para a **memória flash** (com a variação emocional de cada mensagem).
5. **Em background** (`core/pipeline`), sem atrasar a resposta: `notetaker` (notas pessoais),
   `summary` (resumo, estado, cenário) e `consolidator` (events + knowledge). Rodam a cada N
   mensagens novas e, ao fim da conversa, com `finalize`.

Turnos da mesma persona são serializados. Se um turno falha, as emoções voltam ao que eram e nada é gravado.

## Organização

| Pasta | Conteúdo |
|---|---|
| `.configs/engine/` | um JSON por componente (`actor`, `primary_interpreter`, `notetaker`, `summary`, `consolidator`, `emotions`, `*_memory`) e o dicionário de emoções |
| `.configs/prompts/` | um `.md` por modelo |
| `.configs/providers/` | um JSON por provedor de LLM, formato OpenAI-compatível (`ollama.json` local, `groq.json` nuvem) |
| `.data/global.db` | registro de personas (uuid, nome, status, última conversa, metadados) |
| `.data/personas/<uuid>/` | `descriptions/`, `parameters.json`, `emotions.json`, `state.md`, `scene.json`, `summary.md`, `assets/` e `memories/` |
| `.data/personas/<uuid>/memories/` | um banco SQLite por memória: `messages.db`, `events.db`, `knowledge.db`, `notes.db` |
| `.logs/` | `general.log` + `<categoria>/<classe>.log`; o conteúdo dos prompts fica em `<classe>.prompt.log`, fora do log geral |

`persona_example` é o modelo copiado ao criar uma persona. Os bancos e logs são gerados em execução e ignorados pelo git.

## Modelos: local (Ollama) ou nuvem (Groq)

Todos os componentes usam o mesmo provider, escolhido por `PERSONA_PROVIDER` (padrão `ollama`).
Copie `.env.example` para `.env` ou exporte as variáveis.

| | Local | Nuvem |
|---|---|---|
| `PERSONA_PROVIDER` | `ollama` | `groq` |
| Pré-requisito | `ollama serve` e `ollama pull llama3.2` | `GROQ_API_KEY` ([console.groq.com/keys](https://console.groq.com/keys)) |
| Modelo padrão | `llama3.2` | `llama-3.3-70b-versatile` |

O modelo padrão fica em `default_model` no JSON do provider. Para usar outro em um componente,
adicione ao `.configs/engine/<componente>.json` `"models": {"groq": "...", "ollama": "..."}`
(ou `"model": "..."` para valer em qualquer provider). Os modelos precisam seguir instruções de
formato JSON (interpretador, notetaker, summary e consolidator respondem em JSON): modelos muito
pequenos podem errar; nesses casos o sistema usa o fallback ou repete o lote depois.

## Conversando

### CLI

```
python -m src.cli --new "Ana"     # cria a persona a partir de persona_example
python -m src.cli --list          # lista as personas (mais recentes primeiro)
python -m src.cli <uuid>          # conversa: /status mostra emoções e resumo, /sair encerra
```

Ao sair, o CLI processa o que o background ainda não processou ("salvando memórias...").
Edite as descrições em `.data/personas/<uuid>/descriptions/`; elas são relidas a cada turno.

### API HTTP

```
pip install -r requirements.txt
python -m src.api                 # http://127.0.0.1:8000  (documentação interativa em /docs)
```

| Rota | O que faz |
|---|---|
| `GET/POST /v1/personas`, `GET/PATCH /v1/personas/{id}` | menu: listar (`status`, `search`), criar, renomear, arquivar |
| `POST /v1/personas/{id}/chat` | `{"messages": ["oi"], "conversation_id": null}` → `{"reply": "..."}` |
| `POST /v1/personas/{id}/finalize` | processa agora o que falta no background |
| `GET /v1/personas/{id}/status` | emoções atuais, resumo, estado e cenário |
| `GET/PUT .../config/parameters` | parâmetros (o `name` fica sincronizado com o menu) |
| `GET/PUT .../config/descriptions` | personalidade, traços, aparência |
| `GET/POST/DELETE .../config/assets` | imagens da persona (png, jpeg, gif, webp, até 5 MB) |
| `GET/POST/DELETE .../config/memory/{events,knowledge,notes}` | listar/buscar (`q`), criar e apagar |
| `GET/DELETE .../config/memory/messages` | mensagens recentes e apagar |

**Sem autenticação**: o servidor escuta só em `127.0.0.1` por padrão. Para expor na rede, ponha um
proxy com autenticação na frente. Só ids de personas registradas chegam ao disco, e os uploads têm
nome gerado e tipo detectado pelo conteúdo (SVG não é aceito).

## Memórias

| Memória | Guarda | Busca |
|---|---|---|
| flash | últimas k mensagens, com a variação emocional de cada uma | direta; o prompt do actor a ajusta ao espaço que sobra |
| events | o que aconteceu (factual, sem impressões) | FTS5 (bm25, sem acentos), com embeddings opcionais |
| knowledge | o que a persona aprendeu, como relações entre entidades (graph-RAG) | acha entidades pelo nome e expande a vizinhança |
| notes | impressões pessoais | igual a events |

Events, knowledge e notes guardam data de criação e de último acesso. Para somar busca semântica a
events e notes, defina `"embedding_model"` em `.configs/engine/events_memory.json` e
`notes_memory.json` (por exemplo um modelo de embeddings do Ollama; o provider precisa expor
`/embeddings`). Sem isso, ou se o embedder falhar, usa só a busca lexical.

## Emoções

Roda de Plutchik: 8 emoções primárias (cada uma em 3 intensidades) e 24 díades (amor, culpa,
deleite...). Cada emoção decai em direção ao temperamento (`baseline`) da persona, em tempo real, com
meia-vida por emoção: `x(t) = base + (x0 - base) * 2^(-dt / meia_vida)`. Emoções opostas se inibem.
Nomes e descrições ficam em `.configs/engine/emotions_description.json`; os parâmetros, em
`.configs/engine/emotions.json`; o estado e o temperamento, em `emotions.json` da persona.

## Background

| Processo | Gatilho padrão | Escreve em |
|---|---|---|
| notetaker | a cada 6 mensagens | `notes.db` (só trechos com variação emocional relevante) |
| summary | a cada 8 | `summary.md`, `state.md`, `scene.json` |
| consolidator | a cada 10 | `events.db`, `knowledge.db` |

Cada um guarda o seu marcador no `messages.db` e só o avança quando termina com sucesso: falha de
rede refaz o lote depois; saída inválida do modelo 3 vezes seguidas descarta aquele lote.
Todos se ajustam (ou se desligam com `"enabled": false`) no JSON do componente.

## Desenvolvimento

```
pip install -r requirements-dev.txt
python -m pytest                  # ou: python -m unittest discover -s tests
```

Python >= 3.11.
