Você consolida a memória de longo prazo de {{name}} a partir de um trecho da conversa. Responda SOMENTE com um objeto JSON válido, sem texto antes ou depois e sem blocos de código.

# Trecho da conversa
{{messages}}

# Memórias parecidas que já existem (com ids)
## Eventos
{{events}}

## Conhecimento
{{knowledge}}

# Formato da resposta
{"events": [{"text": "..."}],
 "knowledge": [{"subject": "...", "relation": "...", "object": "...", "subject_type": "pessoa", "object_type": "lugar"}],
 "obsolete": ["E12", "F7"]}

# Regras
- "events": o que ACONTECEU, de forma factual, sem opiniões nem sentimentos de {{name}} (isso vai para as notas). Terceira pessoa, datas absolutas (AAAA-MM-DD) e frases autossuficientes: nada de "ele", "ontem" ou "hoje". Chame o usuário de "o usuário" se ele não disser o nome.
- "knowledge": o que {{name}} APRENDEU, como relações curtas (sujeito, relação, objeto), com nomes próprios completos e a relação em minúsculas ("mora em", "é irmã de", "gosta de"). Os tipos ("pessoa", "lugar", "coisa", "conceito"...) são opcionais.
- Não repita o que já existe nas memórias parecidas. Se algo mudou ou foi corrigido, adicione o novo e liste o id antigo em "obsolete".
- Só registre o que o trecho realmente diz. Use listas vazias se não houver nada novo.
