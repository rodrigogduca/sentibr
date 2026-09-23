# P1 · 06 — Tarefas

Legenda: `[ ]` pendente · `[x]` feito (critério verificado) · ⏱ estimativa · ✅ critério de aceite · ↔ requisito

## Fase 0 — Setup (⏱ 20 min)
- [ ] **T0.1** Criar o repositório público `sentibr` no GitHub, com `.gitignore` Python + `data/`, `models/`, `.env`
  ✅ repositório acessível sem login
- [x] **T0.2** Criar a estrutura de pastas de `04-design.md`, `requirements.txt` com versões fixadas, e `src/config.py`
  ✅ `pip install -r requirements.txt` sem erro; `python -c "import src.config"` ok
- [ ] **T0.3** Primeiro commit com `specs/` já incluído
  ✅ commit datado visível no GitHub (evidência de autoria)

## Fase 1 — Dados (⏱ 50 min)
- [x] **T1.1** Baixar o Olist via `kagglehub` e salvar em `data/raw/`
  ✅ o arquivo existe e tem cerca de 99 mil linhas
- [x] **T1.2** Implementar `src/dados.py` (limpar, rotular, deduplicar, dividir)
  ✅ `python -m src.dados` gera os 3 CSVs; os splits não têm textos em comum (assert)
- [x] **T1.3** `tests/test_dados.py`: rótulo 1,2→neg, 3→neu, 4,5→pos; limpeza; ausência de duplicatas entre splits
  ✅ `pytest` verde ↔ RNF-4
- [x] **T1.4** *(feito como script `src/eda.py`, reprodutível)* `notebooks/01_eda.ipynb`: distribuição de classes, tamanho dos textos, palavras mais frequentes por classe
  ✅ 3 gráficos salvos em `reports/figuras/`; `reports/dados.md` preenchido

## Fase 2 — Baseline (⏱ 60 min)
- [x] **T2.1** E1 `DummyClassifier` + E2 TF-IDF + LogReg com grid em C
  ✅ `models/baseline.joblib` salvo; F1-macro na validação impresso
- [x] **T2.2** `src/inferencia.py` com `ClassificadorBaseline`
  ✅ `prever(["produto ótimo"])` retorna um dict cujas probabilidades somam 1 ↔ CA-1.1
- [x] **T2.3** `src/avaliar.py`: métricas no teste + matriz de confusão + latência
  ✅ `reports/metricas.json` e `matriz_confusao_baseline.png` gerados; F1-macro ≥ 0,65 ↔ CA-3.1, CA-3.3

## Fase 3 — App (⏱ 60 min) — *vem antes do BERT para garantir o MVP*
- [x] **T3.1** Aba "Texto" com validação de texto vazio ↔ CA-1.1, CA-1.2
  ✅ `streamlit run app/app.py` classifica "Chegou quebrado, péssimo" como negativo
- [x] **T3.2** Aba "Lote (CSV)" com download e gráfico ↔ CA-2.1, CA-2.2, CA-2.3
  ✅ o CSV de exemplo `app/exemplo.csv` (10 linhas) é processado; um CSV sem a coluna `texto` mostra erro
- [x] **T3.3** Aba "Sobre o modelo" lendo `metricas.json`
  ✅ métricas visíveis na interface
- [ ] **T3.4** Commit + push. **Neste ponto o MVP está pronto.**

## Fase 4 — BERTimbau (⏱ 60–90 min, Colab)
- [x] **T4.1** *(treinado na GPU local, RTX 3050 6 GB, via `src/treinar_bert.py`; `notebooks/02_bertimbau_colab.ipynb` reproduz no Colab)* `notebooks/02_bertimbau.ipynb`: carregar os splits (upload ou clone do repositório), tokenizar, `Trainer`
  ✅ treino de 2 épocas termina em ≤ 30 min na T4 ↔ CA-4.2
- [ ] **T4.2** *(modelo salvo em `models/bertimbau/`; falta publicar no HF Hub — opcional)* Salvar o modelo no HF Hub (`push_to_hub`) ou no Drive
  ✅ o modelo carrega do zero em outra sessão
- [x] **T4.3** *(BERTimbau ponderado: F1-macro 0,721 vs 0,682 do baseline, +0,039)* `ClassificadorBert` + seletor no app + BERT incluído no `avaliar.py`
  ✅ `metricas.json` tem as duas entradas; o seletor funciona ↔ CA-1.3, CA-3.1

## Fase 5 — Análise e documentação (⏱ 45 min)
- [x] **T5.1** `reports/erros.md` com ≥ 10 erros categorizados ↔ CA-3.2
- [x] **T5.2** README completo: problema, dados + licença, como rodar, tabela de resultados, figura, limitações, autor
  ✅ outra pessoa (ou uma sessão limpa) reproduz o baseline seguindo só o README ↔ CA-4.1
- [x] **T5.3** `ruff check .` sem erros ↔ RNF-4
- [ ] **T5.4** Seguir `07-entrega.md`
