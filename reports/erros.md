# Análise de erros — BERTimbau (cross-entropy ponderada)

Amostra: 40 erros sorteados (seed 42) do melhor modelo no conjunto de teste, gerada por
`python -m src.avaliar` em [`erros_amostra.csv`](erros_amostra.csv). Cada erro foi lido e classificado
manualmente em uma categoria.

## Resumo

| Categoria | Erros | % | O que significa |
|-----------|------:|--:|-----------------|
| Sentimento misto | 12 | 30% | O texto elogia uma coisa e critica outra ("entrega rápida, mas a qualidade não é boa") |
| Ruído de rótulo | 10 | 25% | A nota do cliente contradiz o texto; o modelo acerta o sentimento e "erra" o rótulo |
| Reclamação implícita | 6 | 15% | Fato negativo descrito sem palavras negativas ("Recebi somente a grelha") |
| Fronteira entre classes vizinhas | 4 | 10% | O texto cabe tanto em neutro quanto na classe ao lado |
| Texto curto/ambíguo | 4 | 10% | Pouca informação ("kkkkk", "Dentro das expectativas") |
| Negação/contrafactual | 2 | 5% | A negação não indica reclamação, ou a crítica vem em forma hipotética |
| Erro claro do modelo | 2 | 5% | O texto é inequívoco e o modelo errou |

**35 dos 40 erros (88%) envolvem a classe neutra**, como rótulo verdadeiro ou como previsão. Só 5 são
confusões diretas entre negativo e positivo, e 4 delas são ruído de rótulo.

Matriz da amostra (linhas = rótulo, colunas = previsão):

| | negativo | neutro | positivo |
|---|---:|---:|---:|
| **negativo** | — | 12 | 1 |
| **neutro** | 6 | — | 2 |
| **positivo** | 4 | 15 | — |

## Exemplos comentados

| # | Texto (trecho) | Rótulo → Previsto | Categoria | Comentário |
|---|----------------|-------------------|-----------|------------|
| 1 | "Cartuchos falsos. Não recomendo…" | positivo → negativo | Ruído de rótulo | Texto claramente negativo com nota 4–5. O modelo está certo. |
| 2 | "Tudo dentro do esperado" | negativo → positivo | Ruído de rótulo | Nada no texto justifica nota 1–2. |
| 3 | "A película era de péssima qualidade e já veio com vários riscos" | neutro → negativo | Ruído de rótulo | Reclamação explícita com nota 3. |
| 4 | "Ok. Pena que não vi as medidas… Não atendeu as expectativas." | positivo → neutro | Ruído de rótulo | Texto negativo, nota alta; o modelo ficou no meio. |
| 5 | "Controle bom. Mas demora muito a entrega." | positivo → neutro | Sentimento misto | Produto elogiado, entrega criticada. |
| 6 | "Produto Frágil. Lindas peças, porém muito frágeis." | negativo → neutro | Sentimento misto | Estética positiva, durabilidade negativa. |
| 7 | "Gostei. Recebi o produto certinho, mas quero o reembolso do frete… sacanagem" | negativo → neutro | Sentimento misto | Começa positivo e termina com raiva. O fim pesou pouco para o modelo. |
| 8 | "Chegou no prazo, porém… veio outro modelo. Mas mesmo assim estou satisfeita" | positivo → neutro | Sentimento misto | Três viradas na mesma frase. |
| 9 | "Recebi somente a Grella oriental. Não recebi as duas panelas." | negativo → neutro | Reclamação implícita | Pedido incompleto, sem adjetivo negativo. |
| 10 | "Bem diferente da foto" | negativo → neutro | Reclamação implícita | Exige conhecimento de mundo: "diferente da foto" é problema. |
| 11 | "Demorou muito o prazo de entrega… Espero que… seja corrigido" | negativo → neutro | Reclamação implícita | Reclamação educada; o tom cordial puxa para neutro. |
| 12 | "o produto não foi recebido ainda por falta minha." | positivo → negativo | Negação | "Não foi recebido" é associado a reclamação, mas o cliente assume a culpa. |
| 13 | "Se realmente tivessem preocupação com os clientes não deixariam a desejar…" | negativo → neutro | Contrafactual | Crítica em forma hipotética, sem palavras negativas diretas. |
| 14 | "kkkkkkkkkkkkkk" | neutro → positivo | Curto/ambíguo | Riso pode ser deboche ou satisfação; sem contexto, é indecidível. |
| 15 | "Como faço para devolver o produto," | neutro → negativo | Curto/ambíguo | Pergunta, não avaliação. Devolução costuma indicar problema. |
| 16 | "Produto entregue entes do prazo." | positivo → neutro | Erro do modelo | Positivo claro; o erro de digitação ("entes") pode ter atrapalhado a tokenização. |
| 17 | "Bom. Gostei do produto comprei de presente" | positivo → neutro | Erro do modelo | Positivo inequívoco. |

## Conclusões

1. **O teto é o rótulo, não só o modelo.** Em 25% da amostra, o rótulo derivado da nota está errado.
   Parte dos "erros" é o modelo discordando corretamente do rótulo. Um conjunto de teste revisado à mão
   daria uma medida mais honesta.
2. **O neutro é, em boa parte, "misto".** Para 30% dos erros, uma única etiqueta por comentário não
   descreve o texto. A evolução natural é a **análise de sentimento por aspecto** (entrega, produto,
   atendimento), que está fora do escopo deste projeto.
3. **O transformer resolve o que o bag-of-words não resolve, mas não tudo.** O BERTimbau acerta
   negações simples ("não gostei"), mas ainda erra reclamações implícitas e contrafactuais, que exigem
   conhecimento de mundo.
4. **Uso recomendado: triagem com revisão humana.** Confusões diretas entre negativo e positivo são raras
   (5 de 40, a maioria ruído de rótulo). Priorizar comentários previstos como negativos é seguro. Decidir
   sobre pessoas com base no modelo não é.
