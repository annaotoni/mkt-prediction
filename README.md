# MKT Sales Prediction

Modelo de regressão linear para prever vendas com base em investimentos publicitários nos canais YouTube, Facebook e Newspaper.

## Stack

- Python 3.12+
- scikit-learn — modelo de regressão linear
- pandas / numpy — manipulação de dados
- matplotlib / seaborn — visualizações
- Flask — API e frontend web

## Estrutura

```
mkt-prediction/
├── app.py                # API Flask + servidor web
├── mkt_prediction.py     # Análise exploratória e treinamento standalone
└── templates/
    └── index.html        # Frontend Bootstrap 5
```

## Como rodar

### Pré-requisitos

```bash
pip install flask pandas numpy matplotlib seaborn scikit-learn
```

### Script de análise (gera gráficos e métricas)

```bash
python mkt_prediction.py
```

### Aplicação web

```bash
python app.py
```

Acesse `http://localhost:5000` — insira os valores de investimento por canal e veja a previsão de vendas em tempo real.

> **Dados reais:** se tiver o arquivo `MKT.csv` original, descomente a linha `data = pd.read_csv("MKT.csv")` em ambos os arquivos e remova o bloco de geração sintética. O CSV deve ter as colunas `youtube`, `facebook`, `newspaper` e `sales`.

## Endpoint da API

```
POST /predict
Content-Type: application/json

{ "youtube": 100, "facebook": 150, "newspaper": 50 }
```

```json
{ "sales": 25.59 }
```

## Modelo

Regressão linear múltipla com divisão 80/20 treino/teste (`random_state=42`).

| Métrica | Valor (dataset sintético) |
|---|---|
| R² | ~0.03 |
| MSE | ~18.5 |

> O R² baixo reflete o dataset sintético. Com os dados reais o modelo original atingia performance significativamente superior.
