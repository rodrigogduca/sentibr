# P1 · 04 — Design

## Arquitetura
```
data/raw/*.csv ──► src/dados.py ──► data/processed/{treino,validacao,teste}.csv
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               ▼                                                   ▼
   src/baseline.py (local, CPU)                 src/treinar_bert.py (GPU local ou Colab)
   TF-IDF + LogisticRegression                  fine-tuning neuralmind/bert-base-portuguese-cased
               │                                                   │
   models/baseline.joblib                       models/bertimbau/ (opcional: HF Hub)
               └───────────────► src/avaliar.py ◄──────────────────┘
                                 reports/metricas.json, matriz_confusao_*.png, erros.md
                                         │
                                         ▼
                                  app/app.py (Streamlit) ── usa src/inferencia.py
```

## Stack
| Camada | Tecnologia | Motivo |
|--------|------------|--------|
| Dados | pandas, kagglehub | padrão, rápido |
| Baseline | scikit-learn (TfidfVectorizer, LogisticRegression, LinearSVC), NLTK | rápido, interpretável |
| DL | PyTorch + Hugging Face `transformers` (`Trainer`), `datasets`, `evaluate` | exigido no edital, com fine-tuning simples |
| Interface | Streamlit | protótipo em poucas horas |
| Qualidade | ruff, pytest | constituição |

## Estrutura de código
```
sentibr/
├── specs/
├── notebooks/
│   ├── 01_eda.ipynb              # exploração: tamanhos de texto, distribuição de classes, nuvem de palavras
│   └── (02_bertimbau → substituído por src/treinar_bert.py, roda em GPU local ou no Colab)
├── src/
│   ├── __init__.py
│   ├── config.py                 # SEED, caminhos, mapa nota→classe, LABELS = ["negativo","neutro","positivo"]
│   ├── dados.py                  # carregar(), limpar(), rotular(), dividir()  → CLI: python -m src.dados
│   ├── baseline.py               # treinar_baseline() com GridSearch pequeno → python -m src.baseline
│   ├── inferencia.py             # classe Classificador (interface comum aos 2 modelos)
│   └── avaliar.py                # métricas + matriz + erros → python -m src.avaliar
├── app/app.py
├── tests/test_dados.py           # rotular(), limpar(), deduplicação
├── models/ (gitignore, exceto README)
├── reports/
├── requirements.txt
└── README.md
```

## Interface comum de inferência (`src/inferencia.py`)
```python
class Classificador(Protocol):
    nome: str
    def prever(self, textos: list[str]) -> list[dict[str, float]]:
        """Retorna, para cada texto, {"negativo": p, "neutro": p, "positivo": p} (soma = 1)."""

class ClassificadorBaseline:  # carrega models/baseline.joblib (Pipeline sklearn)
class ClassificadorBert:      # carrega do HF Hub/pasta local; softmax sobre os logits
```
Com essa interface, o app e o `avaliar.py` tratam os dois modelos do mesmo jeito.

## Decisões de modelagem
- **Baseline:** `Pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50_000, sublinear_tf=True),
  LogisticRegression(class_weight="balanced", max_iter=2000))`. `GridSearchCV` pequeno em `C ∈ {0.5, 1, 2, 4}`
  com `scoring="f1_macro"`, cv=3, usando só o treino. Comparar com LinearSVC calibrado (`CalibratedClassifierCV`).
- **BERTimbau:** `AutoModelForSequenceClassification(num_labels=3)`, lr = 2e-5, batch 32, 2 épocas, warmup 10%,
  weight_decay 0,01, `fp16=True`, melhor checkpoint por F1-macro na validação. Se o tempo apertar, subamostrar o
  treino para 20 mil exemplos, estratificado.
- **Desbalanceamento:** `class_weight="balanced"` no baseline. No BERT, primeiro sem pesos; se a classe neutra
  ficar com recall < 0,2, testar loss ponderada.

## App (`app/app.py`)
- Barra lateral: seletor de modelo, com BERTimbau desativado se não conseguir carregar.
- Aba "Texto": `st.text_area` + botão → classe em destaque + barras de probabilidade.
- Aba "Lote (CSV)": `st.file_uploader` → tabela + gráfico de distribuição + `st.download_button`.
- Aba "Sobre o modelo": métricas de `reports/metricas.json` + matriz de confusão.
- `@st.cache_resource` para carregar cada modelo uma vez só.
