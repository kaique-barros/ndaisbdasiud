# Consolidador de conhecimento (chama o modelo; o prompt é montado em core/consolidator.py).
#
# Diferença para o summary: o summary só mantém o fio da conversa (scene.json / state.md);
# aqui o que foi dito vira memória de longo prazo:
#   - events:    o que ACONTECEU, factual, sem impressões da persona
#   - knowledge: o que a persona APRENDEU (entidades e relações para o graph-rag),
#                com timestamp de criação e de último acesso
# As impressões pessoais continuam no notetaker (notes), não aqui.
#
# Entrada : lote de mensagens ainda não consolidadas + trechos já existentes parecidos
#           (para atualizar/mesclar em vez de duplicar)
# Saída   : JSON com `events`, `knowledge` (novos/atualizados) e `obsolete` (ids substituídos)
# Execução: em background, depois do turno (a cada N mensagens ou ao fim da conversa),
#           idempotente via marcador `consolidated_up_to` (id da última mensagem consolidada).
