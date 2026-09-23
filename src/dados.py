"""Preparação dos dados: carregar → limpar → rotular → deduplicar → dividir.

Uso: python -m src.dados
"""

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DADOS_BRUTOS, DADOS_PROC, NOTA_PARA_CLASSE, RELATORIOS, SEED

_ESPACOS = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    """Remove quebras de linha e espaços repetidos."""
    return _ESPACOS.sub(" ", str(texto)).strip()


def baixar(destino: Path = DADOS_BRUTOS) -> None:
    """Baixa o CSV de avaliações do Olist (Kaggle) se ainda não estiver em data/raw/."""
    if destino.exists():
        return
    import kagglehub

    origem = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(origem / destino.name, destino)


def carregar(caminho: Path = DADOS_BRUTOS) -> pd.DataFrame:
    """Lê o CSV da Olist e monta a coluna `texto` = título + mensagem."""
    df = pd.read_csv(caminho)
    titulo = df["review_comment_title"].fillna("").map(normalizar)
    mensagem = df["review_comment_message"].fillna("").map(normalizar)
    df["texto"] = [
        f"{t}. {m}" if t and m else (t or m) for t, m in zip(titulo, mensagem, strict=True)
    ]
    return df[["review_id", "review_score", "texto"]]


def limpar(df: pd.DataFrame, min_chars: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["texto"] = df["texto"].map(normalizar)
    return df[df["texto"].str.len() >= min_chars].reset_index(drop=True)


def rotular(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rotulo"] = df["review_score"].map(NOTA_PARA_CLASSE)
    return df.dropna(subset=["rotulo"]).reset_index(drop=True)


def deduplicar(df: pd.DataFrame) -> pd.DataFrame:
    """Remove textos repetidos (ignorando caixa) para evitar vazamento entre treino e teste."""
    chave = df["texto"].str.lower()
    return df.loc[~chave.duplicated()].reset_index(drop=True)


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado 70/15/15."""
    treino, resto = train_test_split(
        df, test_size=0.30, stratify=df["rotulo"], random_state=SEED
    )
    validacao, teste = train_test_split(
        resto, test_size=0.50, stratify=resto["rotulo"], random_state=SEED
    )
    return treino, validacao, teste


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows (cp1252)
    baixar()
    bruto = carregar()
    limpo = limpar(bruto)
    rotulado = rotular(limpo)
    unico = deduplicar(rotulado)
    treino, validacao, teste = dividir(unico)

    DADOS_PROC.mkdir(parents=True, exist_ok=True)
    cols = ["review_id", "texto", "rotulo"]
    for nome, parte in [("treino", treino), ("validacao", validacao), ("teste", teste)]:
        parte[cols].to_csv(DADOS_PROC / f"{nome}.csv", index=False)

    comuns = set(treino["texto"].str.lower()) & set(teste["texto"].str.lower())
    assert not comuns, "Vazamento: textos em comum entre treino e teste"

    linhas = [
        "# Relatório de dados (gerado por `python -m src.dados`)",
        "",
        "| Etapa | Linhas |",
        "|-------|--------|",
        f"| Bruto | {len(bruto):,} |",
        f"| Com texto (≥ 3 caracteres) | {len(limpo):,} |",
        f"| Rotulado | {len(rotulado):,} |",
        f"| Após deduplicação | {len(unico):,} |",
        "",
        "## Distribuição de classes por split",
        "",
        "| Split | Linhas | negativo | neutro | positivo |",
        "|-------|--------|----------|--------|----------|",
    ]
    for nome, parte in [("treino", treino), ("validacao", validacao), ("teste", teste)]:
        prop = parte["rotulo"].value_counts(normalize=True)
        linhas.append(
            f"| {nome} | {len(parte):,} | {prop.get('negativo', 0):.1%} | "
            f"{prop.get('neutro', 0):.1%} | {prop.get('positivo', 0):.1%} |"
        )
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    (RELATORIOS / "dados.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
