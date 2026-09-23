# P1 · 05 — Experimentos e Avaliação

## Hipótese
Um transformer pré-treinado em português (BERTimbau) captura negação, ironia leve e contexto melhor que
bag-of-words, ganhando pelo menos 3 pontos de F1-macro sobre TF-IDF + Regressão Logística, a um custo de
inferência cerca de 10x maior.

## Protocolo
1. Os splits são fixados uma vez (`src/dados.py`, seed 42) e reutilizados por todos os modelos.
2. Hiperparâmetros são escolhidos **só** com treino + validação.
3. O teste é avaliado **uma única vez** por modelo final.
4. Tudo é registrado em `reports/metricas.json`:
   ```json
   {"baseline_logreg": {"f1_macro": 0.0, "acuracia": 0.0, "por_classe": {...}, "latencia_ms": 0.0},
    "bertimbau":      {...}}
   ```

## Métricas
| Métrica | Papel |
|---------|-------|
| **F1-macro** | principal (trata as 3 classes igualmente, apesar do desbalanceamento) |
| F1 / recall por classe | diagnóstico, principalmente do neutro |
| Acurácia | contexto (enganosa com classes desbalanceadas) |
| Matriz de confusão | figura do relatório |
| Latência média por texto (CPU) | custo × benefício |

## Experimentos planejados (ordem = prioridade)
| ID | Experimento | Obrigatório? |
|----|-------------|--------------|
| E1 | Baseline majoritário (`DummyClassifier`) — piso de referência | sim |
| E2 | TF-IDF + LogReg (grid em C) | sim |
| E3 | TF-IDF + LinearSVC calibrado | se sobrar 15 min |
| E4 | BERTimbau fine-tuning 2 épocas | sim |
| E5 | Variante binária (sem neutro) para os melhores modelos | extra |

## Análise de erros (obrigatória, `reports/erros.md`)
Separar pelo menos 10 erros do melhor modelo e classificar cada um em:
- **Ruído de rótulo** (o texto é positivo e a nota é 1, por exemplo);
- **Negação/ironia** ("não é que seja ruim...");
- **Texto curto/ambíguo** ("ok", "chegou");
- **Sentimento misto** ("produto ótimo, entrega péssima") — mostra por que a análise por aspecto seria uma evolução.

## Vieses, limitações e ética
- **Domínio:** treinado em e-commerce, então pode errar em outros contextos (saúde, política). O app avisa isso.
- **Linguagem:** gírias regionais e erros de digitação estão sub-representados, e o desempenho pode variar
  por região/escolaridade do autor. Não é possível medir isso aqui (não há dados demográficos, o que é correto
  pela LGPD). Fica registrado como limitação.
- **Uso indevido:** o modelo não deve ser usado para decidir sobre pessoas (ex.: punir atendentes). O uso
  previsto é triagem com revisão humana.
- **Ruído de rótulo:** a métrica tem um teto, porque parte dos "erros" é erro do rótulo.

## Critérios de parada (timebox)
- Se o BERTimbau não terminar até 22/09 às 12h, entregar o baseline completo e registrar o BERT como
  "em andamento" no README. Não bloqueia a declaração: o projeto já é NLP supervisionado.
