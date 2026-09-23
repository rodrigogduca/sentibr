import pandas as pd

from src.dados import deduplicar, dividir, limpar, normalizar, rotular


def _df(textos, notas):
    return pd.DataFrame({"review_id": range(len(textos)), "review_score": notas, "texto": textos})


def test_normalizar_remove_quebras_e_espacos():
    assert normalizar("  ótimo\n\nproduto   recomendo ") == "ótimo produto recomendo"


def test_rotular_mapeia_notas_para_classes():
    df = rotular(_df(["a", "b", "c", "d", "e"], [1, 2, 3, 4, 5]))
    assert df["rotulo"].tolist() == ["negativo", "negativo", "neutro", "positivo", "positivo"]


def test_limpar_remove_textos_curtos():
    df = limpar(_df(["ok", "  ", "bom demais"], [5, 5, 5]))
    assert df["texto"].tolist() == ["bom demais"]


def test_deduplicar_ignora_caixa():
    df = deduplicar(_df(["Ótimo", "ótimo", "ruim"], [5, 5, 1]))
    assert len(df) == 2


def test_dividir_sem_textos_em_comum_e_proporcoes():
    textos = [f"texto numero {i}" for i in range(1000)]
    notas = [1 if i % 5 == 0 else 3 if i % 5 == 1 else 5 for i in range(1000)]
    treino, validacao, teste = dividir(rotular(_df(textos, notas)))
    assert len(treino) == 700 and len(validacao) == 150 and len(teste) == 150
    assert not set(treino["texto"]) & set(teste["texto"])
    assert not set(treino["texto"]) & set(validacao["texto"])
