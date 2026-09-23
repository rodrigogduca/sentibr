# P1 · 01 — Visão e Escopo

## Nome
**SentiBR: classificação de sentimento em avaliações de clientes em português**

## Problema
Empresas de varejo recebem milhares de avaliações em texto livre. Ler tudo à mão é inviável. Reclamações
urgentes se perdem no meio de elogios, e a nota de 1 a 5 nem sempre reflete o que o texto diz. É preciso triar
automaticamente o **sentimento** de cada comentário para priorizar o atendimento e medir a satisfação.

## Usuário-alvo
Analista de atendimento ou CX de um e-commerce ou pequeno negócio (ex.: cafeteria, loja virtual) que quer
saber, em segundos, se um comentário é **negativo, neutro ou positivo**.

## Proposta de valor
- Classifica um comentário (ou um CSV com vários) em 3 classes, com probabilidade.
- Compara duas abordagens, a **clássica** (TF-IDF + Regressão Logística) e a **deep learning** (BERTimbau),
  mostrando o ganho real do transformer e o custo dele.

## Alinhamento com o Edital 032/2026 (Anexo I, item 1)
| Temática exigida | Onde aparece no projeto |
|------------------|-------------------------|
| Algoritmos supervisionados, scikit-learn | Baseline TF-IDF + LogisticRegression / LinearSVC |
| DL, PyTorch, transformers | Fine-tuning do BERTimbau com Hugging Face |
| Análise de sentimentos, classificação de texto, spaCy/NLTK | Tarefa central + pré-processamento |
| Vieses algorítmicos, impactos sociais | Análise de erros e de viés em `05-experimentos.md` |
| Coleta/limpeza/transformação de dados | Pipeline de preparo em `03-dados.md` |

## Escopo (dentro)
- Dataset público Olist (avaliações reais, anonimizadas).
- 3 classes: negativo (notas 1–2), neutro (3), positivo (4–5).
- Baseline clássico + 1 transformer ajustado.
- App Streamlit: texto único e upload de CSV.
- README, relatório técnico em PDF, declaração.

## Fora de escopo
- Detecção de aspectos ("entrega", "produto") — possível evolução.
- Deploy em nuvem com escalabilidade, API autenticada, monitoramento em produção.
- Coleta de dados próprios ou de redes sociais (evita LGPD e prazo).
- Idiomas além do português.

## Critério de sucesso do projeto
1. Baseline com **F1-macro ≥ 0,65** no teste (3 classes; a classe neutra é difícil e isso fica registrado).
2. BERTimbau com **F1-macro ≥ baseline + 0,03**.
3. App funcionando localmente com os dois modelos (ou só o baseline, se o transformer não couber no tempo).
4. Repositório público reprodutível + PDF + declaração até 23/09.

## Timebox
Cerca de 5 h até o MVP (22/09, manhã).
