# Plataforma Teste

Análise de redes financeiras com grafos inteligentes.

## Funcionalidades

- Upload CSV + feature engineering automático
- Grafos por co-ocorrência com pesos
- Detecção de comunidades (Leiden, Louvain, etc.)
- Visualização interativa (vis.js) e Sankey
- Exportação JSON + HTML + ZIP

## Como Executar

```bash
.venv/bin/python -m streamlit run app/dashboard.py
```

## Estrutura

app/dashboard.py - Interface principal
app/graph_builder.py - Construção de grafos
app/graph_engine/ - Motor de análise
