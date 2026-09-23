"""Avalia todos os modelos disponíveis no MESMO conjunto de teste.

Gera reports/metricas.json, matrizes de confusão e reports/erros_amostra.csv.
Uso: python -m src.avaliar
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from src.config import (  # noqa: E402
    BERT_DIR,
    BERT_SEM_PESOS_DIR,
    DADOS_PROC,
    FIGURAS,
    LABELS,
    RELATORIOS,
    SEED,
)
from src.inferencia import (  # noqa: E402
    Classificador,
    ClassificadorBaseline,
    ClassificadorBert,
    classe_prevista,
)


def latencia_ms(clf: Classificador, textos: list[str], n: int = 50) -> float:
    """Latência média por texto, prevendo um texto por vez."""
    inicio = time.perf_counter()
    for t in textos[:n]:
        clf.prever([t])
    return (time.perf_counter() - inicio) / n * 1000


def resumo(y_true, y_pred) -> dict:
    rel = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "acuracia": round(accuracy_score(y_true, y_pred), 4),
        "por_classe": {
            c: {k: round(rel[c][k], 4) for k in ("precision", "recall", "f1-score")} for c in LABELS
        },
    }


def salvar_matriz(y_true, y_pred, chave: str, titulo: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABELS, normalize="true", values_format=".2f", ax=ax, cmap="Blues"
    )
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(FIGURAS / f"matriz_confusao_{chave}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    treino = pd.read_csv(DADOS_PROC / "treino.csv")
    teste = pd.read_csv(DADOS_PROC / "teste.csv")
    textos, y_true = teste["texto"].tolist(), teste["rotulo"].tolist()
    FIGURAS.mkdir(parents=True, exist_ok=True)

    metricas: dict[str, dict] = {}

    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
    dummy.fit(treino["texto"], treino["rotulo"])
    metricas["dummy_majoritario"] = resumo(y_true, dummy.predict(textos))

    modelos: dict[str, Classificador] = {"baseline_logreg": ClassificadorBaseline()}
    if BERT_SEM_PESOS_DIR.exists():
        modelos["bertimbau_sem_pesos"] = ClassificadorBert(BERT_SEM_PESOS_DIR)
        modelos["bertimbau_sem_pesos"].nome = "BERTimbau sem pesos (ablação)"
    if BERT_DIR.exists():
        modelos["bertimbau"] = ClassificadorBert()

    predicoes: dict[str, list[str]] = {}
    for chave, clf in modelos.items():
        probs = clf.prever(textos)
        y_pred = [classe_prevista(p) for p in probs]
        predicoes[chave] = y_pred
        metricas[chave] = resumo(y_true, y_pred)
        salvar_matriz(y_true, y_pred, chave, clf.nome)

    # Latência sempre em CPU, para comparar os dois modelos em condição de uso comum.
    metricas["baseline_logreg"]["latencia_ms_cpu"] = round(
        latencia_ms(modelos["baseline_logreg"], textos), 2
    )
    if "bertimbau" in modelos:
        metricas["bertimbau"]["latencia_ms_cpu"] = round(
            latencia_ms(ClassificadorBert(dispositivo="cpu"), textos), 2
        )

    RELATORIOS.mkdir(parents=True, exist_ok=True)
    (RELATORIOS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    # Amostra de erros do melhor modelo, para a análise qualitativa (reports/erros.md).
    melhor = max(predicoes, key=lambda k: metricas[k]["f1_macro"])
    teste["previsto"] = predicoes[melhor]
    erros = teste[teste["rotulo"] != teste["previsto"]]
    erros.sample(n=min(40, len(erros)), random_state=SEED)[
        ["texto", "rotulo", "previsto"]
    ].to_csv(RELATORIOS / "erros_amostra.csv", index=False)
    print(f"Amostra de erros do modelo '{melhor}' salva em reports/erros_amostra.csv")


if __name__ == "__main__":
    main()
