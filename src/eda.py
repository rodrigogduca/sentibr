"""Análise exploratória: distribuição de classes, tamanho dos textos e termos por classe.

Uso: python -m src.eda
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

from src.baseline import stopwords_pt  # noqa: E402
from src.config import DADOS_PROC, FIGURAS, LABELS  # noqa: E402

CORES = {"negativo": "#d62728", "neutro": "#bcbd22", "positivo": "#2ca02c"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(DADOS_PROC / "treino.csv")
    FIGURAS.mkdir(parents=True, exist_ok=True)

    contagem = df["rotulo"].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(contagem.index, contagem.values, color=[CORES[c] for c in LABELS])
    for i, v in enumerate(contagem.values):
        ax.text(i, v, f"{v / contagem.sum():.0%}", ha="center", va="bottom")
    ax.set_title("Distribuição de classes (treino)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_classes.png", dpi=120)
    plt.close(fig)

    df["palavras"] = df["texto"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for c in LABELS:
        ax.hist(df.loc[df["rotulo"] == c, "palavras"].clip(upper=80), bins=40, alpha=0.5,
                label=c, color=CORES[c], density=True)
    ax.set_xlabel("palavras por comentário (cortado em 80)")
    ax.set_title("Tamanho dos comentários por classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_tamanho.png", dpi=120)
    plt.close(fig)

    vec = CountVectorizer(stop_words=stopwords_pt(), ngram_range=(1, 2), min_df=5,
                          strip_accents=None)
    X = vec.fit_transform(df["texto"].str.lower())
    termos = vec.get_feature_names_out()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, c in zip(axes, LABELS, strict=True):
        freq = X[(df["rotulo"] == c).to_numpy()].sum(axis=0).A1
        top = pd.Series(freq, index=termos).nlargest(12)[::-1]
        ax.barh(top.index, top.values, color=CORES[c])
        ax.set_title(f"Termos mais frequentes — {c}")
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_termos.png", dpi=120)
    plt.close(fig)

    print(df.groupby("rotulo")["palavras"].describe()[["mean", "50%", "max"]].round(1))
    print(f"Comentários com mais de 100 palavras: {(df['palavras'] > 100).mean():.1%}")


if __name__ == "__main__":
    main()
