"""Interface comum de inferência para o baseline e o BERTimbau."""

from pathlib import Path
from typing import Protocol

import joblib

from src.config import BASELINE_PATH, BERT_DIR, BERT_MAX_LEN, LABELS


class Classificador(Protocol):
    nome: str

    def prever(self, textos: list[str]) -> list[dict[str, float]]:
        """Para cada texto, retorna {"negativo": p, "neutro": p, "positivo": p} (soma = 1)."""
        ...


def classe_prevista(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


class ClassificadorBaseline:
    nome = "Baseline (TF-IDF + LogReg)"

    def __init__(self, caminho: Path = BASELINE_PATH):
        self.modelo = joblib.load(caminho)
        self.classes = list(self.modelo.classes_)

    def prever(self, textos: list[str]) -> list[dict[str, float]]:
        probs = self.modelo.predict_proba(textos)
        return [{c: float(p[self.classes.index(c)]) for c in LABELS} for p in probs]


class ClassificadorBert:
    def __init__(self, caminho: Path = BERT_DIR, dispositivo: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(caminho)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(caminho)
        self.modelo.to(self.dispositivo).eval()
        self.id2label = self.modelo.config.id2label
        self.nome = "BERTimbau (fine-tuned)"

    def prever(self, textos: list[str], lote: int = 64) -> list[dict[str, float]]:
        saida: list[dict[str, float]] = []
        for i in range(0, len(textos), lote):
            enc = self.tokenizer(
                textos[i : i + lote],
                truncation=True,
                max_length=BERT_MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(self.dispositivo)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.modelo(**enc).logits, dim=-1).cpu().numpy()
            for p in probs:
                por_rotulo = {self.id2label[j]: float(v) for j, v in enumerate(p)}
                saida.append({c: por_rotulo[c] for c in LABELS})
        return saida
