"""SentiBR — interface Streamlit.

Uso: streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estilo import CSS, carimbo, tira  # noqa: E402

from src.config import BASELINE_PATH, BERT_DIR, FIGURAS, LABELS, RELATORIOS  # noqa: E402
from src.inferencia import ClassificadorBaseline, ClassificadorBert, classe_prevista  # noqa: E402

st.set_page_config(page_title="SentiBR", page_icon="💬", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def carregar_baseline() -> ClassificadorBaseline:
    return ClassificadorBaseline()


@st.cache_resource
def carregar_bert() -> ClassificadorBert:
    return ClassificadorBert()


st.markdown(
    '<div class="sb-topo"><h1>SentiBR</h1>'
    '<div class="sub">Sentimento em avaliações de clientes · PT-BR</div></div>'
    '<hr class="sb-perf">',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Modelo")
    opcoes = []
    if BASELINE_PATH.exists():
        opcoes.append("Baseline (TF-IDF + LogReg)")
    if BERT_DIR.exists():
        opcoes.append("BERTimbau (fine-tuned)")
    if not opcoes:
        st.error("Nenhum modelo treinado. Rode `python -m src.baseline`.")
        st.stop()
    escolha = st.radio("Escolha o modelo", opcoes, index=len(opcoes) - 1)
    st.info(
        "Treinado com avaliações de e-commerce. Pode errar em outros domínios. "
        "Use para triagem, com revisão humana."
    )

clf = carregar_bert() if escolha.startswith("BERT") else carregar_baseline()

aba_texto, aba_lote, aba_sobre = st.tabs(["Texto", "Lote (CSV)", "Sobre o modelo"])

with aba_texto:
    texto = st.text_area("Comentário do cliente", placeholder="Ex.: Chegou antes do prazo, adorei!")
    if st.button("Classificar", type="primary"):
        if not texto.strip():
            st.warning("Digite um comentário antes de classificar.")
        else:
            probs = clf.prever([texto])[0]
            classe = classe_prevista(probs)
            st.markdown(carimbo(classe, probs[classe]), unsafe_allow_html=True)
            st.markdown(tira({c: probs[c] for c in LABELS}), unsafe_allow_html=True)

with aba_lote:
    st.write("Envie um CSV com a coluna **`texto`**.")
    arquivo = st.file_uploader("Arquivo CSV", type="csv")
    if arquivo is not None:
        df = pd.read_csv(arquivo)
        if "texto" not in df.columns:
            st.error("O CSV precisa ter uma coluna chamada `texto`.")
        else:
            df["texto"] = df["texto"].fillna("").astype(str)
            probs = clf.prever(df["texto"].tolist())
            df["sentimento"] = [classe_prevista(p) for p in probs]
            for c in LABELS:
                df[f"prob_{c}"] = [round(p[c], 4) for p in probs]
            st.bar_chart(df["sentimento"].value_counts().reindex(LABELS, fill_value=0))
            st.dataframe(df, width="stretch")
            st.download_button(
                "Baixar resultado",
                df.to_csv(index=False).encode("utf-8"),
                file_name="sentimentos.csv",
                mime="text/csv",
            )

with aba_sobre:
    caminho = RELATORIOS / "metricas.json"
    if caminho.exists():
        metricas = json.loads(caminho.read_text(encoding="utf-8"))
        tabela = pd.DataFrame(
            {
                nome: {
                    "F1-macro": m["f1_macro"],
                    "Acurácia": m["acuracia"],
                    "F1 neutro": m["por_classe"]["neutro"]["f1-score"],
                    "Latência CPU (ms)": m.get("latencia_ms_cpu"),
                }
                for nome, m in metricas.items()
            }
        ).T
        st.dataframe(tabela)
        for figura in sorted(FIGURAS.glob("matriz_confusao_*.png")):
            st.image(str(figura))
    else:
        st.write("Rode `python -m src.avaliar` para gerar as métricas.")

st.markdown(
    '<hr class="sb-perf">'
    '<div class="sb-rodape">modelo de triagem &middot; requer revisão humana<br>'
    'treinado em avaliações de e-commerce (2016&ndash;2018)</div>',
    unsafe_allow_html=True,
)
