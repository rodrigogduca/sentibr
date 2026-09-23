Modelos treinados (não versionados). Gere com:
- `python -m src.baseline` → `baseline.joblib`
- `python -m src.treinar_bert` → `bertimbau/` (cross-entropy ponderada, modelo principal)
- `python -m src.treinar_bert --sem-pesos --saida models/bertimbau_sem_pesos` → ablação
