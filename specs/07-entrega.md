# P1 · 07 — Entrega e Comprovação

## Artefatos
| Artefato | Local | Obrigatório até 23/09? |
|----------|-------|------------------------|
| Repositório público com código, specs e README | `github.com/rodrigogduca/sentibr` | sim |
| Métricas e figuras | `reports/` no repositório | sim |
| Relatório técnico em PDF (2–4 págs) | `reports/relatorio-tecnico-p1.pdf` | sim |
| Declaração assinada da iniciativa estudantil | PDF separado (fora do repositório) | **sim — é o comprovante do B.5** |
| Modelo BERTimbau publicado | HF Hub | desejável |
| Screenshot do app funcionando | `reports/figuras/app.png` (vai no README e no PDF) | sim |

## Checklist de comprovação (item B.5 do barema)
- [ ] O README tem nome do autor, período e a temática do edital atendida (NLP, DL, transformers)
- [ ] O relatório PDF foi preenchido a partir de `_compartilhado/modelos/relatorio-tecnico.md`, com o resultado real
- [ ] A declaração foi preenchida a partir de `_compartilhado/modelos/declaracao-iniciativa.md`, com:
  - título: **"SentiBR: classificação de sentimento em avaliações de clientes em português"**
  - temática: **"Processamento de Linguagem Natural e Deep Learning (transformers)"**
  - link do repositório e resultado principal (F1-macro)
- [ ] A declaração está assinada (digital, ou assinada à mão e escaneada), legível, em PDF ≤ 10 MB
- [ ] Nome do arquivo: `B5_P1_Declaracao_SentiBR.pdf`
- [ ] Anexada no formulário de inscrição, campo "g" (critérios classificatórios)

## Lattes (reforço, não substitui o comprovante)
Produção técnica → Programa de computador sem registro: "SentiBR — classificação de sentimento em PT-BR com
BERTimbau", 2026, com o link do repositório.

## Pitch de 30 s para a entrevista
"Construí um classificador de sentimento para avaliações em português. Comecei com um baseline TF-IDF com
regressão logística, que chegou a F1-macro de X, e fiz fine-tuning do BERTimbau, que chegou a Y. A classe
neutra foi a mais difícil, e mostro na análise de erros que parte disso é ruído do próprio rótulo. Também
discuto os limites de domínio e o uso responsável, porque é uma ferramenta de triagem com revisão humana,
não de decisão."
