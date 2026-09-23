"""Treina o baseline TF-IDF + Regressão Logística com busca em grade (só no treino).

Uso: python -m src.baseline
"""

import json

import joblib
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import BASELINE_PATH, DADOS_PROC, NEGACOES, RELATORIOS, SEED


def stopwords_pt() -> list[str]:
    """Stopwords do NLTK em português, mantendo as negações."""
    try:
        palavras = nltk.corpus.stopwords.words("portuguese")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        palavras = nltk.corpus.stopwords.words("portuguese")
    return sorted(set(palavras) - NEGACOES)


def criar_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
            ),
        ]
    )


def main() -> None:
    treino = pd.read_csv(DADOS_PROC / "treino.csv")
    validacao = pd.read_csv(DADOS_PROC / "validacao.csv")

    grade = {
        "clf__C": [0.5, 1, 2, 4],
        "tfidf__stop_words": [None, stopwords_pt()],
        "tfidf__strip_accents": [None, "unicode"],
    }
    busca = GridSearchCV(criar_pipeline(), grade, scoring="f1_macro", cv=3, n_jobs=-1, verbose=1)
    busca.fit(treino["texto"], treino["rotulo"])

    modelo = busca.best_estimator_
    f1_val = f1_score(validacao["rotulo"], modelo.predict(validacao["texto"]), average="macro")

    params = {
        "C": busca.best_params_["clf__C"],
        "stopwords": busca.best_params_["tfidf__stop_words"] is not None,
        "strip_accents": busca.best_params_["tfidf__strip_accents"],
        "f1_macro_cv": round(busca.best_score_, 4),
        "f1_macro_validacao": round(f1_val, 4),
    }
    print(json.dumps(params, indent=2, ensure_ascii=False))

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, BASELINE_PATH)
    (RELATORIOS / "baseline_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
