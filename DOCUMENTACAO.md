# Documentação técnica — SentiBR (P1)

Guia completo do projeto: o que é, como foi construído, como reproduzir do zero, o que cada arquivo e
cada função faz, por que cada decisão foi tomada, o que deu errado no caminho e como defender o
trabalho numa entrevista.

Leitura rápida: [`README.md`](README.md). Especificação formal: [`specs/`](specs/).
Este documento é o meio-termo: mais detalhado que o README, mais prático que as specs.

---

## Sumário

1. [O problema em uma página](#1-o-problema-em-uma-página)
2. [Visão geral da arquitetura](#2-visão-geral-da-arquitetura)
3. [Instalação passo a passo](#3-instalação-passo-a-passo)
4. [Execução passo a passo](#4-execução-passo-a-passo)
5. [Os dados, etapa por etapa](#5-os-dados-etapa-por-etapa)
6. [O baseline, linha por linha](#6-o-baseline-linha-por-linha)
7. [O BERTimbau, linha por linha](#7-o-bertimbau-linha-por-linha)
8. [Avaliação: como os números são produzidos](#8-avaliação-como-os-números-são-produzidos)
9. [O aplicativo Streamlit](#9-o-aplicativo-streamlit)
10. [Testes, lint e CI](#10-testes-lint-e-ci)
11. [Referência de arquivos e funções](#11-referência-de-arquivos-e-funções)
12. [Decisões técnicas e alternativas descartadas](#12-decisões-técnicas-e-alternativas-descartadas)
13. [Resultados completos](#13-resultados-completos)
14. [Problemas comuns (troubleshooting)](#14-problemas-comuns-troubleshooting)
15. [Perguntas de entrevista e respostas](#15-perguntas-de-entrevista-e-respostas)
16. [Limitações, ética e próximos passos](#16-limitações-ética-e-próximos-passos)

---

## 1. O problema em uma página

**Contexto.** Uma loja online recebe milhares de comentários por mês. Ler tudo é inviável; ignorar é
pior. A pergunta prática é: *quais comentários precisam de atenção agora, e como está a satisfação ao
longo do tempo?*

**Tarefa de ML.** Classificação de texto em três classes (`negativo`, `neutro`, `positivo`), com a
probabilidade de cada uma. É aprendizado **supervisionado**, **multiclasse**, **de rótulo único**.

**Por que três classes e não uma nota de 1 a 5?** Porque a decisão operacional é ternária: triar
(negativo), ignorar (positivo) ou olhar depois (neutro). Prever a nota exata seria mais difícil e menos
útil — o erro entre 4 e 5 não muda nada para quem atende o cliente.

**De onde vêm os rótulos.** Da nota que o próprio cliente deu (1–2 → negativo, 3 → neutro, 4–5 →
positivo). Isso é **supervisão fraca** (*weak supervision*): não é um anotador humano lendo o texto, é
um sinal correlacionado e barato. A consequência aparece na seção 13: parte do "erro" do modelo é, na
verdade, ruído do rótulo.

**Métrica principal: F1-macro.** As classes são desbalanceadas (62% positivo, 29% negativo, 9%
neutro). A acurácia premia quem chuta "positivo" sempre: o chute majoritário já dá 62,4% de acurácia e
é inútil. O F1-macro é a média simples do F1 das três classes, então a classe rara pesa tanto quanto a
comum, e quem ignora o neutro é punido.

$$F1_{macro} = \frac{1}{3}\left(F1_{neg} + F1_{neu} + F1_{pos}\right)$$

**Meta definida ANTES dos experimentos** (`specs/05-experimentos.md`): o transformer precisa superar o
baseline em pelo menos **+0,03 de F1-macro** para justificar o custo computacional. Resultado real:
**+0,039**. A meta escrita antes é o que impede a racionalização depois ("0,005 já é ganho").

---

## 2. Visão geral da arquitetura

```
                        ┌───────────────────────────────────────────┐
                        │ Kaggle: olist_order_reviews_dataset.csv   │
                        │ 99.224 avaliações (2016–2018)             │
                        └────────────────────┬──────────────────────┘
                                             │ kagglehub
                                             ▼
   src/dados.py    carregar → limpar → rotular → deduplicar → dividir (70/15/15, seed 42)
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     ▼                       ▼                       ▼
              treino (25.495)        validação (5.463)         teste (5.464)
                     │                       │                       │
                     │  ajusta               │  escolhe              │  só no fim,
                     │  parâmetros           │  hiperparâmetros      │  uma vez
                     │                       │                       │
      ┌──────────────┴───────────┐           │                       │
      ▼                          ▼           │                       │
src/baseline.py            src/treinar_bert.py                       │
TF-IDF + LogReg            fine-tuning do BERTimbau                  │
GridSearchCV (cv=3)        2 épocas, CE ponderada                    │
      │                          │                                   │
      ▼                          ▼                                   │
models/baseline.joblib    models/bertimbau/                          │
      │                          │                                   │
      └────────────┬─────────────┘                                   │
                   ▼                                                 │
          src/inferencia.py — interface comum                        │
          prever(textos) → [{"negativo": p, "neutro": p, ...}]       │
                   │                                                 │
       ┌───────────┴───────────┐                                     │
       ▼                       ▼                                     ▼
  app/app.py            src/avaliar.py ◄─────────────────────────────┘
  (Streamlit)           métricas · matrizes · latência · amostra de erros
                               │
                               ▼
                        reports/metricas.json
                        reports/figuras/*.png
                        reports/erros_amostra.csv → reports/erros.md (análise manual)
```

**A peça central é `src/inferencia.py`.** Os dois modelos implementam o mesmo `Protocol`
(`Classificador`), com o método `prever(textos) -> list[dict]`. Por causa disso, o app e o avaliador
não sabem — nem precisam saber — qual modelo está ativo. Trocar de modelo é trocar um objeto.

---

## 3. Instalação passo a passo

### 3.1 Pré-requisitos

| Item | Versão | Observação |
|------|--------|-----------|
| Python | 3.11 ou superior | `python --version` |
| Espaço em disco | ~4 GB | pesos do BERTimbau (~440 MB) + torch (~2,5 GB) |
| GPU NVIDIA | opcional | só para *treinar* o BERT; a inferência roda em CPU |
| Conta no Kaggle | sim | o `kagglehub` pede login na primeira execução |

Sem GPU, há duas saídas: treinar no Google Colab com o notebook
[`notebooks/02_bertimbau_colab.ipynb`](notebooks/02_bertimbau_colab.ipynb) (T4 gratuita, ~20 min), ou
rodar só o baseline, que treina em CPU em cerca de 2 minutos.

### 3.2 Ambiente virtual

```bash
git clone https://github.com/rodrigogduca/sentibr.git
cd sentibr

python -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell: .venv\Scripts\Activate.ps1)
source .venv/bin/activate        # Linux / macOS
```

> **Por que ambiente virtual?** O projeto fixa versões exatas em `requirements.txt`. Instalar no Python
> do sistema mistura essas versões com as de outros projetos e quebra a reprodutibilidade — que é
> justamente o que um portfólio de ML precisa provar.

### 3.3 Dependências

**Com GPU NVIDIA** (instale o torch CUDA *antes*, senão o pip resolve para a versão CPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Só CPU:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verifique se a GPU foi reconhecida:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# esperado numa máquina com GPU: True NVIDIA GeForce RTX 3050 ...
```

### 3.4 O que cada dependência faz

| Pacote | Para quê |
|--------|----------|
| `pandas`, `numpy` | manipulação dos CSVs e dos vetores |
| `scikit-learn` | TF-IDF, regressão logística, `GridSearchCV`, métricas, matriz de confusão |
| `nltk` | lista de stopwords em português (testada na busca em grade) |
| `torch` | backend do transformer |
| `transformers` | BERTimbau, tokenizador, `Trainer` |
| `accelerate` | exigido pelo `Trainer` para gerenciar dispositivo e precisão mista |
| `joblib` | serialização do pipeline do baseline |
| `kagglehub` | download do dataset da Olist |
| `matplotlib` | figuras (EDA, matrizes de confusão) |
| `streamlit` | interface web |
| `pytest`, `ruff` | testes e lint |

---

## 4. Execução passo a passo

A ordem importa: cada etapa consome o que a anterior produziu.

### Passo 1 — Preparar os dados (~1 min)

```bash
python -m src.dados
```

O que acontece:
1. `baixar()` — se `data/raw/olist_order_reviews_dataset.csv` não existir, baixa o dataset da Olist com
   o `kagglehub` e copia só o arquivo de avaliações. Na primeira vez o Kaggle pede autenticação.
2. `carregar()` — lê o CSV e concatena `review_comment_title` + `review_comment_message` na coluna
   `texto`.
3. `limpar()` — normaliza espaços e descarta textos com menos de 3 caracteres (a maioria das 99 mil
   avaliações é só nota, sem comentário).
4. `rotular()` — aplica `NOTA_PARA_CLASSE`.
5. `deduplicar()` — remove textos repetidos ignorando maiúsculas.
6. `dividir()` — split estratificado 70/15/15 com `random_state=42`.
7. Um `assert` verifica que não há texto em comum entre treino e teste. Se houver, o programa **para**.
8. Escreve `data/processed/{treino,validacao,teste}.csv` e o relatório `reports/dados.md`.

Saída esperada:

```
| Etapa | Linhas |
| Bruto | 99,224 |
| Com texto (≥ 3 caracteres) | 42,377 |
| Após deduplicação | 36,422 |
| treino | 25,495 | 28.6% | 9.0% | 62.4% |
```

### Passo 2 — Exploração (~30 s)

```bash
python -m src.eda
```

Gera três figuras em `reports/figuras/`: distribuição das classes (`eda_classes.png`), distribuição do
tamanho dos textos (`eda_tamanho.png`, que justifica `max_length=128` tokens) e termos mais frequentes
por classe (`eda_termos.png`).

> **Por que um script e não um notebook?** Um script roda no CI, não guarda estado escondido entre
> células e produz sempre a mesma figura. A spec previa `notebooks/01_eda.ipynb`; a troca por
> `src/eda.py` está registrada em `specs/06-tarefas.md` (T1.4).

### Passo 3 — Treinar o baseline (~2 min em CPU)

```bash
python -m src.baseline
```

Roda `GridSearchCV` com 16 combinações × 3 folds = 48 treinos, **apenas sobre o conjunto de treino**.
Salva `models/baseline.joblib` e `reports/baseline_params.json`.

Saída real deste projeto:

```json
{ "C": 0.5, "stopwords": false, "strip_accents": "unicode",
  "f1_macro_cv": 0.6744, "f1_macro_validacao": 0.6804 }
```

Repare: a busca escolheu **não** remover stopwords. Essa é uma descoberta, não um descuido — veja a
seção 12.

### Passo 4 — Fine-tuning do BERTimbau (~10 min em RTX 3050, ~20 min em T4)

```bash
python -m src.treinar_bert                    # modelo principal (cross-entropy ponderada)
```

Opções:

| Flag | Padrão | Para quê |
|------|--------|----------|
| `--epocas` | 2 | número de passagens pelo treino |
| `--lote` | 16 | batch de treino (reduza para 8 se faltar VRAM) |
| `--amostra N` | 0 (tudo) | subamostra estratificada, útil para testar o código rápido |
| `--sem-pesos` | desligado | desliga a ponderação de classes (é a ablação) |
| `--saida` | `models/bertimbau` | pasta de destino |

A ablação, que gera a linha "sem pesos" da tabela de resultados:

```bash
python -m src.treinar_bert --sem-pesos --saida models/bertimbau_sem_pesos
```

### Passo 5 — Avaliar tudo no mesmo teste (~3 min em CPU)

```bash
python -m src.avaliar
```

Avalia **todos** os modelos presentes em `models/` sobre o mesmo conjunto de teste, e escreve:
- `reports/metricas.json` — F1-macro, acurácia, precisão/recall/F1 por classe, latência;
- `reports/figuras/matriz_confusao_*.png` — uma por modelo, normalizada por linha;
- `reports/erros_amostra.csv` — 40 erros sorteados do melhor modelo, para a análise manual.

### Passo 6 — Análise de erros (manual)

Abrir `reports/erros_amostra.csv`, ler os 40 casos, classificar cada um em uma categoria e escrever
`reports/erros.md`. **Esta etapa é humana de propósito** — é o que separa "rodei um modelo" de
"entendi o problema". O resultado está na seção 13.

### Passo 7 — Interface

```bash
streamlit run app/app.py
```

Abre em `http://localhost:8501`.

### Passo 8 — Verificação

```bash
pytest -q          # 5 testes
ruff check .       # lint
```

### Resumo dos comandos

```bash
python -m src.dados        # 1. dados        → data/processed/*.csv, reports/dados.md
python -m src.eda          # 2. exploração   → reports/figuras/eda_*.png
python -m src.baseline     # 3. baseline     → models/baseline.joblib
python -m src.treinar_bert # 4. transformer  → models/bertimbau/
python -m src.avaliar      # 5. avaliação    → reports/metricas.json + figuras
streamlit run app/app.py   # 6. app
pytest && ruff check .     # 7. qualidade
```

---

## 5. Os dados, etapa por etapa

### 5.1 Fonte

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
licença **CC BY-NC-SA 4.0** (uso não comercial, com atribuição e compartilhamento igual). São ~100 mil
pedidos reais feitos em marketplaces brasileiros entre 2016 e 2018, já anonimizados pela Olist: não há
nome, e-mail, CPF nem endereço exato. O arquivo usado aqui é `olist_order_reviews_dataset.csv`.

Os dados **não são versionados** no repositório: seriam ~15 MB de conteúdo de terceiros, e o
`.gitignore` exclui `data/`. `src/dados.py` baixa tudo sob demanda, o que mantém o repositório leve e
a reprodução honesta.

### 5.2 Funil de limpeza

| Etapa | Linhas | O que saiu |
|-------|-------:|------------|
| Bruto | 99.224 | — |
| Com texto (≥ 3 caracteres) | 42.377 | 57% das avaliações são só nota, sem comentário |
| Rotulado | 42.377 | nenhuma perda: toda nota está entre 1 e 5 |
| Após deduplicação | 36.422 | 5.955 textos repetidos ("Ótimo", "Recomendo", "Bom") |

### 5.3 Deduplicação: o detalhe que evita uma métrica falsa

Em avaliações de e-commerce, textos curtos se repetem muito. Se "Ótimo produto" aparece 300 vezes e o
split for feito antes de deduplicar, o mesmo texto cai no treino **e** no teste. O modelo memoriza e a
métrica de teste sobe sem que o modelo tenha aprendido nada generalizável — é **vazamento de dados**
(*data leakage*).

```python
def deduplicar(df):
    chave = df["texto"].str.lower()
    return df.loc[~chave.duplicated()].reset_index(drop=True)
```

Duas garantias no código: um `assert` em `main()` e o teste
`test_dividir_sem_textos_em_comum_e_proporcoes`. Uma garantia que só existe no comentário não
sobrevive à próxima refatoração; uma que está num teste, sim.

### 5.4 Split estratificado 70/15/15

```python
treino, resto = train_test_split(df, test_size=0.30, stratify=df["rotulo"], random_state=42)
validacao, teste = train_test_split(resto, test_size=0.50, stratify=resto["rotulo"], random_state=42)
```

- **Estratificado** (`stratify`): cada split preserva a proporção 62/29/9. Sem isso, o neutro (9%)
  poderia ficar mal representado no teste e a métrica viraria loteria.
- **`random_state=42`**: o mesmo split em qualquer máquina. Reprodutibilidade não é detalhe estético;
  sem ela, comparar dois modelos é comparar duas coisas diferentes.
- **Três conjuntos, não dois**: a validação escolhe hiperparâmetros, o teste é aberto uma única vez, no
  fim. Ajustar hiperparâmetros olhando o teste é otimizar contra a régua — a métrica publicada deixa de
  medir generalização.

### 5.5 Por que não há stemming, lematização ou remoção de stopwords fixa

O pré-processamento pesado é herança de quando os modelos eram frágeis. Aqui:
- **stopwords** entraram como *hiperparâmetro* da busca em grade, e a busca escolheu mantê-las;
- **negações** (`não`, `nunca`, `nem`, `jamais`, `sem`...) nunca são removidas, mesmo quando as
  stopwords são: "não gostei" e "gostei" são opostos, e as listas padrão do NLTK incluem "não";
- **stemming/lematização** não foram usados: o TF-IDF com bigramas já captura boa parte da variação, e
  o BERTimbau trabalha com subpalavras (WordPiece), então reduzir palavra a radical antes só destrói
  informação.

---

## 6. O baseline, linha por linha

### 6.1 Por que começar por um baseline

Um baseline simples responde à pergunta que o gestor faz: *o modelo caro vale a pena?* Sem ele,
qualquer número de transformer parece bom. Aqui há dois pisos:

| Piso | F1-macro | Papel |
|------|---------:|-------|
| `DummyClassifier(strategy="most_frequent")` | 0,256 | prova que a acurácia de 62% não significa nada |
| TF-IDF + Regressão Logística | 0,682 | o alvo real a ser superado |

### 6.2 O pipeline

```python
Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                              max_features=50_000, sublinear_tf=True)),
    ("clf",   LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
])
```

Cada parâmetro, e o motivo:

| Parâmetro | Valor | Por quê |
|-----------|-------|---------|
| `ngram_range=(1, 2)` | unigramas + bigramas | "não recomendo" e "muito bom" só existem como bigrama |
| `min_df=2` | termo em ≥ 2 documentos | descarta erros de digitação únicos; reduz o vocabulário |
| `max_features=50_000` | teto do vocabulário | controla memória e overfitting |
| `sublinear_tf=True` | `1 + log(tf)` | repetir "péssimo" 5 vezes não é 5× mais informativo |
| `class_weight="balanced"` | peso ∝ 1/frequência | um erro no neutro custa ~7× um erro no positivo |
| `max_iter=2000` | — | o padrão (100) não converge com 50 mil features |

**Usar um `Pipeline` do scikit-learn não é organização, é correção.** Dentro da validação cruzada, o
TF-IDF é ajustado *só* no fold de treino de cada rodada. Se o TF-IDF fosse ajustado antes, no dataset
inteiro, as estatísticas de frequência do fold de validação vazariam para o vetorizador.

### 6.3 A busca em grade

```python
grade = {
    "clf__C": [0.5, 1, 2, 4],                  # inverso da regularização L2
    "tfidf__stop_words": [None, stopwords_pt()],
    "tfidf__strip_accents": [None, "unicode"],
}
GridSearchCV(pipeline, grade, scoring="f1_macro", cv=3, n_jobs=-1)
```

16 combinações × 3 folds = 48 ajustes, com `scoring="f1_macro"` (a métrica do projeto, não a acurácia
padrão) e `n_jobs=-1` (todos os núcleos). A busca roda **apenas no treino**; a validação serve depois
como conferência independente — `f1_macro_cv` = 0,674 e `f1_macro_validacao` = 0,680 são próximos, o
que indica que a busca não superajustou os folds.

### 6.4 O resultado escolhido

`C=0.5` (a regularização mais forte da grade), **sem** remoção de stopwords, **com** remoção de
acentos. A regularização forte faz sentido com 50 mil features e 25 mil exemplos: há mais dimensões do
que dados, e o modelo precisa ser contido.

---

## 7. O BERTimbau, linha por linha

### 7.1 O modelo

[`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
— BERT-base (12 camadas, 768 dimensões, ~110 M parâmetros) pré-treinado em português do brasil
(brWaC), por Souza, Nogueira e Lotufo (BRACIS, 2020).

**Por que ele e não um BERT multilíngue?** Porque foi pré-treinado só em português, com um vocabulário
WordPiece próprio. O mBERT divide sua capacidade entre 104 idiomas e fragmenta palavras portuguesas em
mais subpalavras. **`cased`** (com maiúsculas) porque "PÉSSIMO" carrega ênfase que "péssimo" não
carrega.

### 7.2 Fine-tuning: o que realmente acontece

Fine-tuning não é treinar do zero. O BERTimbau já sabe português — sintaxe, semântica, contexto. O que
se faz é:
1. adicionar uma cabeça de classificação linear (768 → 3) por cima do token `[CLS]`;
2. treinar **tudo** (corpo + cabeça) com taxa de aprendizado baixa (`2e-5`), para ajustar sem destruir
   o que já foi aprendido — o fenômeno que se evita é o *esquecimento catastrófico*.

### 7.3 Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `learning_rate` | 2e-5 | faixa canônica de fine-tuning de BERT (2e-5 a 5e-5) |
| `num_train_epochs` | 2 | com 25 mil exemplos, 3+ épocas começam a superajustar |
| `per_device_train_batch_size` | 16 | cabe em 6 GB de VRAM com `fp16` |
| `warmup_ratio` | 0.1 | aquece o LR nos primeiros 10% dos passos, evitando um passo destrutivo no início |
| `weight_decay` | 0.01 | regularização L2 padrão do AdamW |
| `max_length` | 128 tokens | cobre >99% das avaliações (ver `eda_tamanho.png`); o custo do atento é quadrático no comprimento |
| `fp16` | se houver CUDA | precisão mista: ~2× mais rápido, metade da memória |
| `load_best_model_at_end` | True | salva a época com melhor F1-macro **de validação**, não a última |
| `metric_for_best_model` | `f1_macro` | a métrica do projeto, de novo |
| `seed` | 42 | reprodutibilidade |

### 7.4 A parte mais importante: cross-entropy ponderada

Esta é a contribuição técnica central do projeto e o assunto que mais rende numa entrevista.

O `Trainer` padrão do `transformers` usa cross-entropy sem pesos. Com 62% de exemplos positivos e 9%
de neutros, a estratégia que minimiza essa perda é **ignorar o neutro**. Foi exatamente o que aconteceu
na primeira versão: acurácia alta (0,866) e recall do neutro de **0,168**.

A correção é uma subclasse de `Trainer`:

```python
class TrainerPonderado(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rotulos = inputs.pop("labels")
        saida = model(**inputs)
        pesos = self.pesos.to(saida.logits.device) if self.pesos is not None else None
        loss = torch.nn.functional.cross_entropy(saida.logits, rotulos, weight=pesos)
        return (loss, saida) if return_outputs else loss
```

Com pesos $w_c = \frac{n}{3 \cdot n_c}$ — o mesmo critério do `class_weight="balanced"` do
scikit-learn. Para este dataset: negativo ≈ 1,17, neutro ≈ 3,71, positivo ≈ 0,53. Errar um neutro custa
7 vezes mais do que errar um positivo.

O efeito, medido no mesmo teste:

| | Acurácia | Recall neutro | F1-macro |
|---|---:|---:|---:|
| Sem pesos | **0,866** | 0,168 | 0,674 |
| Com pesos | 0,847 | **0,466** | **0,721** |

A ponderação **piora a acurácia** e **melhora o que importa**. É o exemplo mais limpo de por que a
métrica precisa ser escolhida antes, e não depois de ver os resultados.

> Detalhe de implementação: a assinatura de `compute_loss` inclui `**kwargs` porque versões recentes do
> `transformers` passam `num_items_in_batch`. Sem isso, o código quebra com `TypeError` ao atualizar a
> biblioteca.

### 7.5 Dataset e colagem

`ConjuntoTexto` tokeniza tudo de uma vez (truncando em 128) e devolve dicionários. O
`DataCollatorWithPadding` aplica **padding dinâmico**: cada batch é preenchido até o maior texto
*daquele batch*, não até 128 fixos. Como a maioria das avaliações é curta, isso corta bastante tempo
de treino sem mudar o resultado.

---

## 8. Avaliação: como os números são produzidos

### 8.1 Regra de ouro

Todos os modelos são avaliados **no mesmo `data/processed/teste.csv`**, que nunca foi usado para
treinar nem para escolher hiperparâmetro nenhum. `src/avaliar.py` detecta os modelos existentes em
`models/` e avalia todos numa única execução — o que torna impossível, por construção, comparar
números gerados em condições diferentes.

### 8.2 O que é calculado

```python
def resumo(y_true, y_pred):
    return {
        "f1_macro": ...,        # métrica principal
        "acuracia": ...,        # contexto (e contraste com o dummy)
        "por_classe": {c: {"precision", "recall", "f1-score"} for c in LABELS},
    }
```

**Matriz de confusão normalizada por linha** (`normalize="true"`): cada linha soma 1 e mostra o
*recall* por classe. Com classes desbalanceadas, a matriz de contagens absolutas engana — 500 erros no
positivo parecem muito e são pouco; 200 no neutro parecem pouco e são metade da classe.

**Latência sempre em CPU**, medindo um texto por vez:

```python
def latencia_ms(clf, textos, n=50):
    inicio = time.perf_counter()
    for t in textos[:n]:
        clf.prever([t])
    return (time.perf_counter() - inicio) / n * 1000
```

Duas escolhas deliberadas: (1) CPU, porque é a condição de uso realista de uma triagem interna, e
porque mediar o BERT na GPU e o baseline na CPU não compararia nada; (2) um texto por vez, porque o
caso de uso interativo é assim — em lote, o BERT fica muito mais rápido por item.

### 8.3 Amostra de erros

```python
melhor = max(predicoes, key=lambda k: metricas[k]["f1_macro"])
erros = teste[teste["rotulo"] != teste["previsto"]]
erros.sample(n=min(40, len(erros)), random_state=42)[...].to_csv("reports/erros_amostra.csv")
```

40 erros do melhor modelo, sorteados com seed fixa — amostra aleatória, não cereja escolhida a dedo.
A leitura e a categorização são manuais, em `reports/erros.md`.

---

## 9. O aplicativo Streamlit

Três abas, mais uma barra lateral para escolher o modelo:

| Aba | O que faz | Requisito |
|-----|-----------|-----------|
| **Texto** | classifica um comentário, mostra a classe, o emoji e as três probabilidades | CA-1.1, CA-1.2 |
| **Lote (CSV)** | recebe um CSV com coluna `texto`, classifica tudo, mostra gráfico e permite baixar o resultado | CA-2.1 a CA-2.3 |
| **Sobre o modelo** | lê `reports/metricas.json` e exibe as métricas reais, sem números escritos à mão | CA-3.x |

Pontos de projeto que valem menção:
- **`@st.cache_resource`** no carregamento do modelo: sem isso, o Streamlit recarregaria os 440 MB do
  BERT a cada interação.
- **Validação de entrada**: texto vazio e CSV sem a coluna `texto` produzem mensagem clara, não
  *stack trace*.
- **Aviso de domínio** visível: o modelo foi treinado em avaliações de e-commerce.
- `app/exemplo.csv` (10 linhas) permite testar o modo lote sem preparar nada.

---

## 10. Testes, lint e CI

### 10.1 Os 5 testes (`tests/test_dados.py`)

| Teste | O que protege |
|-------|---------------|
| `test_normalizar_remove_quebras_e_espacos` | normalização de espaços e quebras de linha |
| `test_rotular_mapeia_notas_para_classes` | a regra 1,2→neg / 3→neu / 4,5→pos |
| `test_limpar_remove_textos_curtos` | o filtro de ≥ 3 caracteres |
| `test_deduplicar_ignora_caixa` | "Ótimo" e "ótimo" contam como o mesmo texto |
| `test_dividir_sem_textos_em_comum_e_proporcoes` | **ausência de vazamento** entre os splits e as proporções 70/15/15 |

Os testes usam **dados sintéticos**: rodam em qualquer máquina, sem Kaggle, sem GPU, em 10 segundos.
O que é testado é a lógica de preparação — a parte onde um erro silencioso invalidaria todas as
métricas do projeto sem levantar nenhuma exceção.

### 10.2 Lint

```bash
ruff check .
```

Configurado em `pyproject.toml`: linha de 100 colunas, alvo `py311`, regras `E` (estilo), `F`
(erros reais), `I` (ordenação de imports), `UP` (sintaxe moderna) e `B` (armadilhas do
`flake8-bugbear`, como argumento padrão mutável).

### 10.3 Integração contínua

`.github/workflows/ci.yml` roda lint e testes a cada push e pull request, em Ubuntu com Python 3.11,
instalando o torch CPU. Como os testes não dependem de dados externos nem de GPU, o CI é rápido e
verde — a diferença entre "passa na minha máquina" e "passa".

---

## 11. Referência de arquivos e funções

### `src/config.py`
Constantes centrais: `SEED=42`, caminhos derivados da raiz do projeto (`Path(__file__).parents[1]`, o
que faz o projeto funcionar de qualquer diretório), `LABELS`, `NOTA_PARA_CLASSE`, o identificador do
BERTimbau, `BERT_MAX_LEN=128` e o conjunto `NEGACOES`.

### `src/dados.py`
| Função | Entrada → saída |
|--------|-----------------|
| `normalizar(texto)` | colapsa espaços e quebras de linha |
| `baixar(destino)` | baixa o CSV da Olist se não existir (idempotente) |
| `carregar(caminho)` | CSV → DataFrame com `review_id`, `review_score`, `texto` |
| `limpar(df, min_chars=3)` | descarta textos curtos |
| `rotular(df)` | nota → classe |
| `deduplicar(df)` | remove textos repetidos (ignorando caixa) |
| `dividir(df)` | 70/15/15 estratificado |
| `main()` | orquestra tudo, valida com `assert` e escreve `reports/dados.md` |

### `src/eda.py`
Três figuras exploratórias em `reports/figuras/`.

### `src/baseline.py`
| Função | Papel |
|--------|-------|
| `stopwords_pt()` | stopwords do NLTK menos as negações (baixa o recurso se faltar) |
| `criar_pipeline()` | TF-IDF + LogReg |
| `main()` | busca em grade, avaliação na validação, salva modelo e parâmetros |

### `src/treinar_bert.py`
| Componente | Papel |
|------------|-------|
| `ConjuntoTexto` | `Dataset` do PyTorch a partir do DataFrame |
| `TrainerPonderado` | `Trainer` com cross-entropy ponderada |
| `pesos_balanceados(rotulos)` | $n / (3 \cdot n_c)$ na ordem de `LABELS` |
| `metricas(pred)` | F1-macro para o `Trainer` |
| `main()` | CLI, treino, avaliação, salvamento, limpeza dos checkpoints |

### `src/inferencia.py`
| Componente | Papel |
|------------|-------|
| `Classificador` (Protocol) | o contrato: `nome` + `prever(textos)` |
| `classe_prevista(probs)` | argmax do dicionário |
| `ClassificadorBaseline` | carrega o joblib, expõe `predict_proba` no formato comum |
| `ClassificadorBert` | tokeniza em lotes de 64, `softmax`, mapeia `id2label` → dicionário |

`ClassificadorBert` importa `torch` e `transformers` **dentro** do `__init__`, não no topo do módulo.
Assim, quem usa só o baseline não paga os ~5 segundos de import do torch.

### `src/avaliar.py`
`latencia_ms`, `resumo`, `salvar_matriz` e `main()`, descritos na seção 8.

### `app/app.py`
Interface Streamlit (seção 9).

### Artefatos gerados

| Caminho | Conteúdo | Versionado? |
|---------|----------|-------------|
| `data/raw/`, `data/processed/` | dados | não (`.gitignore`) |
| `models/baseline.joblib` | pipeline scikit-learn | não |
| `models/bertimbau/` | pesos, tokenizador, config | não |
| `reports/metricas.json` | métricas de todos os modelos | **sim** |
| `reports/dados.md` | funil e distribuição | **sim** |
| `reports/erros.md` + `erros_amostra.csv` | análise de erros | **sim** |
| `reports/figuras/*.png` | EDA, matrizes, screenshot | **sim** |
| `reports/relatorio-tecnico-p1.pdf` | relatório de 2–4 páginas | **sim** |

A regra é simples: **código e resultados entram no git; dados e pesos, não.** Resultados pequenos e em
texto são o que torna o repositório auditável sem inchá-lo.

---

## 12. Decisões técnicas e alternativas descartadas

| Decisão | Alternativa | Por que a escolha |
|---------|-------------|-------------------|
| 3 classes | regressão da nota 1–5 | a decisão operacional é ternária; prever nota exata é mais difícil e menos útil |
| Rótulo derivado da nota | anotação manual | 36 mil exemplos anotados à mão são inviáveis em 1 dia; o custo aparece como teto de métrica, e está documentado |
| F1-macro | acurácia | com 62% de positivos, a acurácia premia o modelo que ignora o neutro |
| Deduplicar antes de dividir | deduplicar depois, ou nunca | evita vazamento e métrica inflada |
| Stopwords como hiperparâmetro | remover sempre | a busca mostrou que remover **piora**: "não", "muito", "mas" carregam sentimento |
| TF-IDF + LogReg como baseline | ir direto ao BERT | sem baseline não há como dizer se o transformer vale o custo |
| Cross-entropy ponderada | `WeightedRandomSampler`; oversampling; focal loss | mesma ideia, implementação mais simples e determinística; e é coerente com o `class_weight` do baseline |
| `max_length=128` | 256 ou 512 | >99% dos textos cabem; o custo do atento cresce com o quadrado do comprimento |
| 2 épocas | 3–5 | a partir da 3ª, o F1-macro de validação estagna e o treino superajusta |
| Interface `Protocol` | `if isinstance(...)` no app | permite trocar de modelo sem tocar no app nem no avaliador |
| Streamlit | FastAPI + front | em um dia, Streamlit entrega interface demonstrável; a API é o próximo passo natural |
| Script de EDA | notebook | roda no CI, sem estado oculto, saída determinística |

---

## 13. Resultados completos

### 13.1 Tabela principal (teste, 5.464 comentários)

| Modelo | F1-macro | Acurácia | F1 neg | F1 neu | F1 pos | Recall neu | Latência CPU |
|--------|---------:|---------:|-------:|-------:|-------:|-----------:|-------------:|
| Dummy (majoritário) | 0,256 | 0,624 | 0,000 | 0,000 | 0,769 | 0,000 | — |
| TF-IDF + LogReg | 0,682 | 0,806 | 0,806 | 0,338 | 0,904 | 0,466 | 1,57 ms |
| BERTimbau sem pesos | 0,674 | **0,866** | 0,858 | 0,229 | **0,935** | 0,168 | — |
| **BERTimbau ponderado** | **0,721** | 0,847 | 0,853 | **0,382** | 0,928 | **0,466** | 73,28 ms |

### 13.2 As três leituras

**1. O transformer vale a pena — com ressalva.** +0,039 de F1-macro sobre o baseline, acima da meta de
+0,03 fixada antes. O custo é 47× mais latência (1,6 ms → 73 ms), ainda muito abaixo do limite de 1 s
por texto definido nos requisitos. Para um volume muito grande e orçamento apertado, o baseline
continua defensável: entrega 95% do resultado por 2% do custo.

**2. A ablação explica o mecanismo.** Sem ponderação, o BERT tem a **maior acurácia da tabela** e um
F1-macro **abaixo do baseline**. Ele aprendeu a jogar o jogo da acurácia: prever quase sempre a classe
comum. Só a métrica correta revela isso.

**3. O neutro é o limite do problema, não do modelo.** Os dois modelos convergem para recall de 0,466
no neutro — número idêntico, por caminhos completamente diferentes. Isso sugere um teto imposto pelos
dados, não pela arquitetura.

### 13.3 O que a análise de erros mostrou

40 erros sorteados, lidos e categorizados à mão (`reports/erros.md`):

| Categoria | Erros | % |
|-----------|------:|--:|
| Sentimento misto | 12 | 30% |
| Ruído de rótulo | 10 | 25% |
| Reclamação implícita | 6 | 15% |
| Fronteira entre classes vizinhas | 4 | 10% |
| Texto curto/ambíguo | 4 | 10% |
| Negação/contrafactual | 2 | 5% |
| **Erro claro do modelo** | **2** | **5%** |

**35 dos 40 erros (88%) envolvem a classe neutra.** E apenas **2 em 40** são erros que qualquer humano
chamaria de erro. Os outros 38 são o problema sendo difícil: o texto elogia e critica ao mesmo tempo
(30%), ou a nota do cliente contradiz o que ele escreveu (25%).

Consequência honesta: **o F1-macro de 0,721 subestima o modelo**, porque uma fração dos "erros" é
acerto contra um rótulo errado. A medida mais justa seria um conjunto de teste reanotado à mão — e
esse é o primeiro item da lista de próximos passos.

---

## 14. Problemas comuns (troubleshooting)

**`CUDA out of memory` durante o treino**
Reduza o batch: `python -m src.treinar_bert --lote 8`. Se persistir, `--lote 4`. Feche outros
programas que usam a GPU (navegadores com aceleração, jogos). A 6 GB de VRAM, lote 16 com `fp16`
funciona, mas com pouca folga.

**`torch.cuda.is_available()` retorna `False` mesmo com GPU NVIDIA**
O torch instalado é a versão CPU. Reinstale na ordem correta:
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**`OSError` / erro de autenticação do `kagglehub`**
É preciso aceitar os termos do dataset e autenticar. Faça login em kaggle.com, baixe `kaggle.json` em
*Settings → API → Create New Token* e coloque em `~/.kaggle/kaggle.json` (Windows:
`C:\Users\<voce>\.kaggle\kaggle.json`).

**`UnicodeEncodeError` no terminal do Windows**
O console usa cp1252 e quebra com acentos. Os scripts já chamam
`sys.stdout.reconfigure(encoding="utf-8")`. Se acontecer em outro script, rode `chcp 65001` antes.

**`LookupError: Resource stopwords not found` (NLTK)**
`stopwords_pt()` baixa o recurso automaticamente na primeira execução. Se a rede estiver bloqueada:
`python -c "import nltk; nltk.download('stopwords')"`.

**O app diz que o modelo não foi encontrado**
Falta rodar `python -m src.baseline` (e/ou `python -m src.treinar_bert`). Os modelos não são
versionados — veja `models/README.md`.

**`TypeError: compute_loss() got an unexpected keyword argument 'num_items_in_batch'`**
Versão do `transformers` mais nova que a assinatura sobrescrita. Já resolvido aqui com `**kwargs` em
`TrainerPonderado.compute_loss`.

**O `GridSearchCV` demora demais**
São 48 ajustes. Com `n_jobs=-1` leva ~2 min num laptop moderno. Para iterar rápido, reduza a grade a
`{"clf__C": [1]}` temporariamente.

**As métricas não batem com as do README**
Confira se `data/processed/` foi gerado com `SEED=42` e se os três CSVs são os mesmos. Mudar a seed
muda o split e, com ele, o terceiro decimal das métricas.

---

## 15. Perguntas de entrevista e respostas

**"Por que F1-macro e não acurácia?"**
As classes são desbalanceadas: 62% positivo, 9% neutro. Um classificador que responde "positivo"
sempre acerta 62,4% — e não serve para nada. O F1-macro dá o mesmo peso às três classes, então ignorar
a classe rara custa caro. Tenho a prova empírica na tabela: o BERT sem pesos tem a maior acurácia do
projeto (0,866) e o segundo pior F1-macro.

**"Como você garantiu que não houve vazamento de dados?"**
Três camadas. Primeiro, deduplicação antes do split — sem isso, "Ótimo produto", que aparece centenas
de vezes, cairia no treino e no teste. Segundo, um `assert` no final de `src/dados.py` que interrompe
a execução se houver texto em comum. Terceiro, um teste automatizado
(`test_dividir_sem_textos_em_comum_e_proporcoes`) que roda no CI. Além disso, o TF-IDF está dentro de
um `Pipeline`, então é ajustado só no fold de treino de cada rodada da validação cruzada.

**"O transformer valeu a pena?"**
Depende do volume. Ganho de +0,039 de F1-macro por 47× mais latência. Para triagem interna, com
volume moderado e 73 ms por texto, vale. Para milhões de textos por dia com orçamento apertado, o
baseline entrega 95% do resultado por 2% do custo. O ponto é que eu tenho os dois números para decidir
— e defini a meta de +0,03 antes de medir, não depois.

**"Qual foi o problema mais difícil?"**
A classe neutra. A primeira versão do BERT tinha recall de 0,168 nela: o modelo aprendeu que ignorar a
classe rara minimiza a perda. Resolvi com cross-entropy ponderada pelo inverso da frequência, numa
subclasse do `Trainer`. O recall subiu para 0,466, a acurácia caiu 2 pontos e o F1-macro subiu 0,047.
Mantive a versão sem pesos como ablação, porque o contraste é o que prova a causa.

**"Se tivesse mais uma semana, o que faria?"**
Reanotaria 500 exemplos de teste à mão. A análise de erros mostra que 25% dos erros são ruído de
rótulo — a nota contradiz o texto — então a métrica atual subestima o modelo e eu não sei por quanto.
Sem essa medida, qualquer melhoria de arquitetura é otimizar contra uma régua torta.

**"Como isso iria para produção?"**
Empacotar `src/inferencia.py` atrás de uma API FastAPI com batching, servir em ONNX Runtime ou com
quantização dinâmica int8 (a latência de CPU cai bastante), monitorar a distribuição das
probabilidades e das classes previstas para detectar *drift*, e manter revisão humana no circuito —
é uma ferramenta de triagem, não de decisão sobre pessoas.

**"Por que o TF-IDF ficou melhor sem remover stopwords?"**
Porque as listas padrão de stopwords foram feitas para recuperação de informação, não para análise de
sentimento. A lista do NLTK em português inclui "não", "muito", "mas" — palavras que carregam
sentimento. Em vez de decidir na intuição, coloquei a remoção como hiperparâmetro na busca em grade, e
a busca escolheu manter. As negações eu protejo explicitamente, via o conjunto `NEGACOES`.

---

## 16. Limitações, ética e próximos passos

### Limitações

- **Domínio.** Treinado em avaliações de e-commerce de 2016–2018. Em saúde, política ou redes sociais
  o desempenho cai — o vocabulário e a forma de expressar insatisfação são outros. O app avisa.
- **Rótulo ruidoso.** ~25% dos erros analisados são casos em que a nota contradiz o texto.
- **Comentários mistos.** ~30% dos erros: "entrega rápida, mas o produto é ruim" não cabe numa única
  etiqueta. O caminho é análise de sentimento **por aspecto**.
- **Viés linguístico.** Gírias regionais e erros de digitação estão sub-representados, e o desempenho
  pode variar com o perfil de quem escreve. Não há dados demográficos para medir isso — o que é
  correto pela LGPD, mas deixa o viés não mensurado.
- **Temporal.** Linguagem muda. Um modelo de 2018 aplicado a 2026 precisa de reavaliação periódica.

### Uso responsável

O sistema é de **triagem com revisão humana**. Não deve ser usado para decidir sobre pessoas — avaliar,
premiar ou punir atendentes — nem para moderação automática sem recurso. Os dados da Olist já vêm
anonimizados; nenhum dado pessoal é coletado, armazenado ou inferido aqui.

### Próximos passos

1. **Reanotar 500 exemplos de teste à mão** para medir o teto real imposto pelo rótulo (maior impacto).
2. **Análise por aspecto** (entrega, produto, atendimento), que resolve a maior categoria de erro.
3. **Calibração de probabilidades** (*temperature scaling*) para que a confiança exibida no app
   signifique o que promete.
4. **Quantização int8 / ONNX** para reduzir a latência de CPU.
5. **API FastAPI** com batching, e monitoramento de drift em produção.
6. **Publicar o modelo no Hugging Face Hub**, para que outra pessoa carregue com uma linha.

---

## Referências

- Souza, F.; Nogueira, R.; Lotufo, R. *BERTimbau: Pretrained BERT Models for Brazilian Portuguese.*
  BRACIS, 2020.
- Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
  NAACL, 2019.
- Olist. *Brazilian E-Commerce Public Dataset*, Kaggle, 2018. CC BY-NC-SA 4.0.
- Documentação: [scikit-learn](https://scikit-learn.org), [Hugging Face Transformers](https://huggingface.co/docs/transformers),
  [Streamlit](https://docs.streamlit.io).

---

*Rodrigo Gandarela Soares de Farias Duca — Engenharia de Computação, Universidade SENAI CIMATEC — setembro de 2026.*
