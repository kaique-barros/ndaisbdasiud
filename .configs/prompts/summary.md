Você mantém o fio da meada da conversa de {{name}}. Atualize o resumo da conversa, o estado atual de {{name}} e o cenário com base no que aconteceu desde a última atualização. Responda SOMENTE com um objeto JSON válido, sem texto antes ou depois e sem blocos de código.

# Resumo atual
{{summary}}

# Estado atual de {{name}}
{{state}}

# Emoções atuais de {{name}}
{{emotions}}

# Cenário atual
{{scene}}

# Mensagens novas
{{messages}}

# Formato da resposta
{"summary": "...", "state": "...", "scene": {"local": "...", "momento": "..."}}

# Regras
- "summary": resumo corrido, em terceira pessoa, que incorpora as mensagens novas ao resumo atual sem perder o essencial (quem disse o quê, decisões, temas). Sem opiniões. No máximo {{max_chars}} caracteres.
- "state": 1 a 3 frases sobre como {{name}} está agora (humor, energia, o que ocupa a cabeça), coerentes com as emoções.
- "scene": objeto com chaves curtas (ex.: local, momento, clima, presentes) e valores curtos. Mantenha as chaves que continuam valendo e atualize as que mudaram.
