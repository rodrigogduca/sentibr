"""Fine-tuning do BERTimbau para classificação de sentimento (3 classes).

Roda em GPU local ou no Google Colab (T4).
Uso: python -m src.treinar_bert [--amostra 20000] [--epocas 2] [--sem-pesos]

Por padrão usa cross-entropy ponderada pelo inverso da frequência das classes (como o
class_weight="balanced" do baseline). Sem pesos, o recall do neutro caiu para 0,17.
"""

import argparse
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import BERT_BASE, BERT_DIR, BERT_MAX_LEN, DADOS_PROC, LABELS, MODELOS, SEED

LABEL2ID = {c: i for i, c in enumerate(LABELS)}


class ConjuntoTexto(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.enc = tokenizer(df["texto"].tolist(), truncation=True, max_length=BERT_MAX_LEN)
        self.rotulos = df["rotulo"].map(LABEL2ID).tolist()

    def __len__(self) -> int:
        return len(self.rotulos)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.rotulos[i]
        return item


class TrainerPonderado(Trainer):
    """Trainer com cross-entropy ponderada, para compensar o desbalanceamento de classes."""

    def __init__(self, *args, pesos: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesos = pesos

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rotulos = inputs.pop("labels")
        saida = model(**inputs)
        pesos = self.pesos.to(saida.logits.device) if self.pesos is not None else None
        loss = torch.nn.functional.cross_entropy(saida.logits, rotulos, weight=pesos)
        return (loss, saida) if return_outputs else loss


def pesos_balanceados(rotulos: pd.Series) -> torch.Tensor:
    """n_total / (n_classes * n_classe), na ordem de LABELS."""
    contagem = rotulos.value_counts()
    return torch.tensor(
        [len(rotulos) / (len(LABELS) * contagem[c]) for c in LABELS], dtype=torch.float
    )


def metricas(pred) -> dict[str, float]:
    y_pred = np.argmax(pred.predictions, axis=-1)
    return {"f1_macro": f1_score(pred.label_ids, y_pred, average="macro")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", type=int, default=0, help="subamostra estratificada do treino")
    parser.add_argument("--epocas", type=float, default=2)
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--sem-pesos", action="store_true", help="cross-entropy sem ponderação")
    parser.add_argument("--saida", default=str(BERT_DIR))
    args = parser.parse_args()

    set_seed(SEED)
    treino = pd.read_csv(DADOS_PROC / "treino.csv")
    validacao = pd.read_csv(DADOS_PROC / "validacao.csv")
    if args.amostra and args.amostra < len(treino):
        treino = treino.groupby("rotulo", group_keys=False).sample(
            frac=args.amostra / len(treino), random_state=SEED
        )

    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )

    checkpoints = MODELOS / "_checkpoints"
    config = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=args.epocas,
        learning_rate=2e-5,
        per_device_train_batch_size=args.lote,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )
    pesos = None if args.sem_pesos else pesos_balanceados(treino["rotulo"])
    print("Pesos das classes:", pesos)
    trainer = TrainerPonderado(
        pesos=pesos,
        model=modelo,
        args=config,
        train_dataset=ConjuntoTexto(treino, tokenizer),
        eval_dataset=ConjuntoTexto(validacao, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metricas,
    )
    trainer.train()
    print("Validação:", trainer.evaluate())

    trainer.save_model(args.saida)
    tokenizer.save_pretrained(args.saida)
    shutil.rmtree(checkpoints, ignore_errors=True)
    print(f"Modelo salvo em {args.saida}")


if __name__ == "__main__":
    main()
