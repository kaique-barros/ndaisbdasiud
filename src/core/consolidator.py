# Relação com engine/base/consolidator.py igual à de core/actor com engine/main/actor:
# aqui se monta o prompt (lote de mensagens ainda não consolidadas + memórias parecidas
# buscadas em paralelo via core/memories) e se aplicam os resultados em events/knowledge.
# Logs: prompt separado do log de eventos, como no actor.
