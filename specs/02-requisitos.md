# P1 · 02 — Requisitos

Formato dos critérios: **EARS** ("QUANDO <gatilho>, O SISTEMA DEVE <resposta>"). Cada critério tem um ID
que é referenciado em `06-tarefas.md`.

## Histórias de usuário

### HU-1 — Classificar um comentário
*Como analista de atendimento, quero colar um comentário e ver o sentimento para priorizar a resposta.*
- **CA-1.1** QUANDO o usuário enviar um texto não vazio, O SISTEMA DEVE exibir a classe prevista
  (negativo/neutro/positivo) e a probabilidade de cada classe.
- **CA-1.2** QUANDO o texto estiver vazio ou só com espaços, O SISTEMA DEVE exibir um aviso e não chamar o modelo.
- **CA-1.3** QUANDO os dois modelos estiverem disponíveis, O SISTEMA DEVE permitir escolher entre "Baseline" e "BERTimbau".

### HU-2 — Classificar em lote
*Como analista, quero enviar um CSV de comentários e baixar o resultado classificado.*
- **CA-2.1** QUANDO o usuário enviar um CSV com a coluna `texto`, O SISTEMA DEVE devolver um CSV com as colunas
  `texto, sentimento, prob_negativo, prob_neutro, prob_positivo`.
- **CA-2.2** QUANDO o CSV não tiver a coluna `texto`, O SISTEMA DEVE exibir uma mensagem de erro clara.
- **CA-2.3** O SISTEMA DEVE exibir um gráfico de barras com a distribuição de sentimentos do lote.

### HU-3 — Entender a qualidade do modelo
*Como avaliador técnico, quero ver as métricas e os erros para confiar no modelo.*
- **CA-3.1** O relatório de avaliação DEVE conter acurácia, precisão, recall e F1 por classe, F1-macro e matriz de
  confusão, para o baseline e para o BERTimbau, no **mesmo** conjunto de teste.
- **CA-3.2** O relatório DEVE listar pelo menos 10 exemplos de erro comentados (categoria do erro).
- **CA-3.3** O baseline DEVE atingir F1-macro ≥ 0,65 e o BERTimbau DEVE superá-lo em ≥ 0,03, ou o README
  registra honestamente o motivo de não ter atingido.

### HU-4 — Reproduzir o experimento
*Como avaliador, quero rodar o projeto do zero.*
- **CA-4.1** QUANDO alguém seguir o README (baixar dados → notebooks/scripts → app), O SISTEMA DEVE reproduzir
  as métricas com diferença ≤ 0,01 no baseline (seed fixa).
- **CA-4.2** O notebook de fine-tuning DEVE rodar do início ao fim no Colab (GPU T4) em ≤ 30 min.

## Requisitos não funcionais
| ID | Requisito |
|----|-----------|
| RNF-1 | Inferência do baseline ≤ 100 ms por texto em CPU; do BERTimbau ≤ 1 s por texto em CPU |
| RNF-2 | Modelos salvos em `models/`: baseline com `joblib` (< 50 MB). BERTimbau no Hugging Face Hub ou Google Drive, **não** no git |
| RNF-3 | Nenhuma informação pessoal nos dados ou exemplos exibidos |
| RNF-4 | Código com `ruff` sem erros; testes de pré-processamento com `pytest` |
| RNF-5 | Interface e documentação em português |
