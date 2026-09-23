# P1 · 03 — Dados

## Fonte
- **Dataset:** Brazilian E-Commerce Public Dataset by Olist (Kaggle: `olistbr/brazilian-ecommerce`)
- **Arquivo usado:** `olist_order_reviews_dataset.csv`
- **Licença:** CC BY-NC-SA 4.0. Uso acadêmico e não comercial permitido, com atribuição. Citar no README.
- **Privacidade:** o dataset já é anonimizado pela Olist (IDs em hash; nomes de lojas substituídos por nomes
  de casas de Game of Thrones).

## Download (reprodutível)
```python
import kagglehub
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")  # requer login Kaggle (token) no Colab
```
Alternativa: download manual pelo site do Kaggle → `data/raw/olist_order_reviews_dataset.csv`.

## Dicionário (colunas relevantes)
| Coluna | Tipo | Uso |
|--------|------|-----|
| `review_id` | str | chave, usada para deduplicação |
| `review_score` | int 1–5 | origem do rótulo |
| `review_comment_title` | str (muito nulo) | concatenado ao texto se existir |
| `review_comment_message` | str (≈ 60% nulo) | **texto de entrada** |
| `review_creation_date` | datetime | só para análise exploratória |

## Rótulo
| Nota | Classe |
|------|--------|
| 1–2 | `negativo` |
| 3 | `neutro` |
| 4–5 | `positivo` |

Proporção esperada após o filtro: cerca de 60% positivo, 30% negativo e 10% neutro. É **desbalanceado**, e a
classe neutra é pequena e ambígua. Por isso a métrica principal é o **F1-macro**, não a acurácia.

## Pipeline de preparo (`src/dados.py`)
1. Carregar o CSV. `texto = título + ". " + mensagem` (ignorando nulos).
2. Remover linhas sem texto ou com texto de menos de 3 caracteres.
3. Normalizar: `strip`, colapsar espaços, remover quebras de linha. Deixar em minúsculas **somente** para o baseline.
4. **Deduplicar pelo texto normalizado** antes do split. Há comentários idênticos ("ótimo", "recomendo"),
   e sem essa etapa eles vazariam entre treino e teste.
5. Criar a coluna `rotulo` conforme a tabela.
6. Split **estratificado** 70/15/15 (treino/validação/teste), `random_state=42`. Salvar em
   `data/processed/{treino,validacao,teste}.csv`.
7. Registrar em `reports/dados.md`: nº de linhas em cada etapa e a distribuição de classes por split.

## Pré-processamento por modelo
- **Baseline:** minúsculas, remoção de acentos opcional (testar os dois), stopwords do NLTK **mantendo negações**
  (`não`, `nunca`, `nem`) porque elas invertem o sentimento; TF-IDF com uni+bigramas.
- **BERTimbau:** texto original (com maiúsculas e acentos). O tokenizer cuida do resto. `max_length=128`
  (mais de 95% dos comentários cabem).

## Riscos de dados
| Risco | Mitigação |
|-------|-----------|
| A nota não reflete o texto (ruído de rótulo) | Aceitar e documentar. Os exemplos entram na análise de erros |
| Neutro ambíguo | F1-macro + análise dedicada; variante binária reportada como extra |
| Vazamento por duplicatas | Deduplicação antes do split (passo 4) |
| Viés de domínio (e-commerce, linguagem informal) | Registrar como limitação: o modelo pode errar mais em outros domínios |
