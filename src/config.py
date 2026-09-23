"""Configurações centrais do projeto SentiBR."""

from pathlib import Path

SEED = 42

RAIZ = Path(__file__).resolve().parents[1]
DADOS_BRUTOS = RAIZ / "data" / "raw" / "olist_order_reviews_dataset.csv"
DADOS_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
RELATORIOS = RAIZ / "reports"
FIGURAS = RELATORIOS / "figuras"

LABELS = ["negativo", "neutro", "positivo"]
NOTA_PARA_CLASSE = {1: "negativo", 2: "negativo", 3: "neutro", 4: "positivo", 5: "positivo"}

BASELINE_PATH = MODELOS / "baseline.joblib"
BERT_BASE = "neuralmind/bert-base-portuguese-cased"
BERT_DIR = MODELOS / "bertimbau"
BERT_SEM_PESOS_DIR = MODELOS / "bertimbau_sem_pesos"  # ablação
BERT_MAX_LEN = 128

# Negações invertem o sentimento, por isso nunca são removidas como stopwords.
NEGACOES = {"não", "nao", "nunca", "nem", "jamais", "nada", "nenhum", "nenhuma", "sem"}
