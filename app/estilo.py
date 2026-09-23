"""Identidade visual do SentiBR: comprovante impresso.

A metáfora é o artefato do mundo do problema. Avaliações de e-commerce chegam
presas a uma compra, e o objeto dessa compra é o comprovante: papel creme,
monoespaçado, régua pontilhada, carimbo. O veredicto é impresso como carimbo,
e não como barra de progresso, porque um carimbo é uma decisão — que é o que
o modelo entrega.
"""

PAPEL = "#FAF6EF"
TINTA = "#22201C"
REGUA = "#D8D0C2"
CORES = {"negativo": "#C4452D", "neutro": "#C08A2E", "positivo": "#3E7C5A"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --papel: #FAF6EF;  --tinta: #22201C;  --regua: #D8D0C2;  --fraco: #7A7367;
  --negativo: #C4452D;  --neutro: #C08A2E;  --positivo: #3E7C5A;
}

html, body, [data-testid="stAppViewContainer"] { background: var(--papel); }
[data-testid="stAppViewContainer"] * { color: var(--tinta); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 760px; padding-top: 2.2rem; }

body, p, li, label, .stMarkdown { font-family: 'Inter', system-ui, sans-serif; }

/* Cabeçalho: o "topo do comprovante" */
.sb-topo { text-align: center; margin-bottom: 4px; }
.sb-topo h1 {
  font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.1rem;
  letter-spacing: -.02em; margin: 0; color: var(--tinta);
}
.sb-topo .sub {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--fraco); margin-top: 6px;
}
.sb-perf { border: 0; border-top: 1.5px dashed var(--regua); margin: 18px 0; }

/* Assinatura da identidade: o carimbo do veredicto */
.sb-carimbo {
  display: inline-block; transform: rotate(-2.5deg); padding: 10px 26px;
  border: 3px double currentColor; border-radius: 3px;
  font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 1.7rem;
  letter-spacing: .1em; text-transform: uppercase; opacity: .92;
}
.sb-carimbo-area { text-align: center; margin: 26px 0 20px; }
.sb-carimbo-area .conf {
  font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: var(--fraco);
  letter-spacing: .1em; margin-top: 14px;
}

/* Tira de probabilidades: linhas de um comprovante, não barras de progresso */
.sb-tira { font-family: 'JetBrains Mono', monospace; font-size: .82rem; margin-top: 6px; }
.sb-linha { display: flex; align-items: center; gap: 10px; padding: 3px 0; }
.sb-linha .rot { width: 78px; text-transform: lowercase; }
.sb-linha .barra { flex: 1; height: 9px; background: #EFE9DE; position: relative; }
.sb-linha .barra i { position: absolute; inset: 0 auto 0 0; display: block; }
.sb-linha .val { width: 54px; text-align: right; font-weight: 500; }

/* Rodapé impresso */
.sb-rodape {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: var(--fraco);
  text-align: center; letter-spacing: .06em; line-height: 1.7;
}

/* Componentes do Streamlit alinhados ao papel */
[data-testid="stSidebar"] { background: #F3EDE2; border-right: 1.5px dashed var(--regua); }
[data-testid="stSidebar"] * { color: var(--tinta); }
.stTextArea textarea {
  background: #FFFDF8; border: 1.5px solid var(--regua); border-radius: 2px;
  font-family: 'Inter', sans-serif; color: var(--tinta);
}
.stTextArea textarea:focus { border-color: var(--tinta); box-shadow: none; }
.stButton button {
  background: var(--tinta); border: 0; border-radius: 2px; padding: .55rem 1.6rem;
}
.stButton button:hover { background: #3C382F; }
/* o rótulo vive num <p> interno: sem isto o texto fica escuro sobre escuro */
.stButton button, .stButton button *, .stButton button p {
  color: var(--papel) !important; font-family: 'JetBrains Mono', monospace;
  font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
}
.stDownloadButton button {
  background: transparent; border: 1.5px solid var(--tinta); border-radius: 2px;
}
.stDownloadButton button, .stDownloadButton button * , .stDownloadButton button p {
  color: var(--tinta) !important; font-family: 'JetBrains Mono', monospace;
  font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
}
[data-baseweb="tab-list"] { gap: 26px; border-bottom: 1.5px dashed var(--regua); }
[data-baseweb="tab"] {
  font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .12em;
  text-transform: uppercase; padding: 8px 0;
}
[data-baseweb="tab-highlight"] { background: var(--tinta); }
/* o fundo do aviso mora no stAlertContainer, não no stAlert */
[data-testid="stAlertContainer"] {
  background: #F7F1E5 !important; border: 0; border-left: 3px solid var(--neutro);
  border-radius: 2px; box-shadow: none;
}
[data-testid="stAlertContainer"] * { color: var(--tinta) !important; font-size: .86rem; }
[data-testid="stAlertContainer"] svg { display: none; }
/* só o rótulo do campo vira etiqueta impressa; as opções do rádio seguem legíveis */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem !important;
  letter-spacing: .12em; text-transform: uppercase; color: var(--fraco) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p {
  font-family: 'Inter', sans-serif; font-size: .88rem !important;
  letter-spacing: 0; text-transform: none; color: var(--tinta) !important;
}
[data-testid="stDataFrame"], [data-testid="stImage"] img { border: 1px solid var(--regua); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""


def carimbo(classe: str, prob: float) -> str:
    """O veredicto impresso como carimbo, com a confiança em monoespaçado."""
    return (
        f'<div class="sb-carimbo-area">'
        f'<div class="sb-carimbo" style="color:{CORES[classe]}">{classe}</div>'
        f'<div class="conf">confiança {prob:.0%}</div>'
        f"</div>"
    )


def tira(probs: dict[str, float]) -> str:
    """Probabilidades como linhas de comprovante: rótulo, traço, valor."""
    linhas = "".join(
        f'<div class="sb-linha"><span class="rot">{c}</span>'
        f'<span class="barra"><i style="width:{p:.1%};background:{CORES[c]}"></i></span>'
        f'<span class="val">{p:.1%}</span></div>'
        for c, p in probs.items()
    )
    return f'<div class="sb-tira">{linhas}</div>'
