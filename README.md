# Forex CSS Stack (MVP -> v1)

Stack Python para replicar o indicador **Currency Slope Strength (CSS)** do MT4 e evoluir para dataset, ML e backtest de estrategia por moeda (nao por par).

## Status atual (entrega 1)

- Estrutura inicial do projeto criada.
- Port do nucleo do indicador:
  - `getSlope` com `ignoreFuture=True` por padrao.
  - `calcCSS` por moeda (8 moedas).
- Download de candles via Twelve Data/OANDA + loader local (`CSV`/`Parquet`).
- Geracao de features para 1 ou N timeframes.
- Dataset multi-timeframe no momento de decisao (8 linhas por timestamp) com targets de basket.
- Treino com walk-forward (baseline + modelo de arvore/boosting).
- Backtest com relatorio (metricas, trades e curva de equity).
- Testes de sanidade iniciais.
- Documento de decisoes tecnicas em [`docs/technical_decisions.md`](docs/technical_decisions.md).
- Resumo do algoritmo MQ4 em [`docs/mq4_algorithm_summary.md`](docs/mq4_algorithm_summary.md).

## Estrutura

```text
data/
  raw/{source}/{pair}/{tf}.parquet|csv
  features/{tf}/
  datasets/
docs/
models/
reports/
scripts/
  download_data.py      # downloader (Twelve Data/OANDA)
  build_features.py     # funcional (MVP atual)
  build_dataset.py      # multi-TF dataset + targets
  train.py              # walk-forward + save model
  backtest.py           # baseline/model backtest
src/forex_css/
  data/
    providers/
  indicators/
  dataset/
  features/
  models/
  backtest/
tests/
```

## Instalacao local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Uso (MVP atual)

### 1) Download (Twelve Data - recomendado para comecar)

Defina a API key no ambiente:

```bash
export TWELVEDATA_API_KEY="SUA_API_KEY"
```

Teste pequeno (recomendado primeiro: 30 dias, 1 par, 2 TF):

```bash
python -m scripts.download_data \
  --provider twelvedata \
  --pairs EURUSD \
  --timeframes H1,D1 \
  --start 2025-01-01 \
  --end 2025-01-31 \
  --data-root data/raw \
  --source twelvedata
```

### 2) Features CSS (multi-TF)

```bash
python -m scripts.build_features \
  --data-root data/raw \
  --source twelvedata \
  --timeframes H1,H4,D1 \
  --pairs AUDCAD,AUDCHF,AUDJPY,AUDNZD,AUDUSD,CADJPY,CHFJPY,EURAUD,EURCAD,EURJPY,EURNZD,EURUSD,GBPAUD,GBPCAD,GBPCHF,GBPJPY,GBPNZD,GBPUSD,NZDCHF,NZDJPY,NZDUSD,USDCAD,USDCHF,USDJPY
```

### 3) Dataset (momento de decisao + targets multi-horizonte)

```bash
python -m scripts.build_dataset \
  --feature-root data/features \
  --timeframes H1,H4,D1 \
  --decision-mode daily \
  --decision-time 21:00 \
  --timezone America/Bahia \
  --data-root data/raw \
  --source twelvedata \
  --target-timeframe H1 \
  --horizons-hours 1,4,8,24 \
  --spread-bps 1.5 \
  --slippage-bps 0.5 \
  --output data/datasets/daily_21brt.parquet
```

### 4) Treino (walk-forward)

```bash
python -m scripts.train \
  --dataset data/datasets/daily_21brt.parquet \
  --model lightgbm \
  --horizon-hours 24 \
  --n-splits 5 \
  --min-train-timestamps 120 \
  --output-model models/model_h24.pkl \
  --report-dir reports/train_h24
```

### 5) Backtest

Baseline (sem ML):

```bash
python -m scripts.backtest \
  --dataset data/datasets/daily_21brt.parquet \
  --horizon-hours 24 \
  --mode baseline \
  --output-dir reports/backtest_baseline \
  --prefix baseline_h24
```

Com modelo treinado:

```bash
python -m scripts.backtest \
  --dataset data/datasets/daily_21brt.parquet \
  --horizon-hours 24 \
  --mode model \
  --model-path models/model_h24.pkl \
  --output-dir reports/backtest_model \
  --prefix model_h24
```

Saidas de backtest:
- `*_trades.csv`
- `*_metrics.json`
- `*_equity.csv`
- `*_equity.png`

## Rodar no Colab (Drive-first)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
```

Fluxo recomendado:

1. Armazenar o projeto e dados no Drive (`/content/drive/MyDrive/forex-css/...`).
2. Instalar dependencias no runtime:

```bash
!pip install -r /content/drive/MyDrive/forex-css/requirements.txt
!pip install -e /content/drive/MyDrive/forex-css
```

3. Rodar os scripts com paths no Drive para persistir artefatos apos restart do runtime.
4. Definir segredo `TWELVEDATA_API_KEY` no Colab (`os.environ["TWELVEDATA_API_KEY"] = "..."`) ou usar Secrets.

## Escolhas tecnicas fixadas

- Dados historicos MVP: **Twelve Data API**
- Opcao de dados via corretora: **OANDA v20** (quando suportado pela jurisdicao da conta)
- Agregacao do basket: **equal-dollar** (media simples das pernas com mesmo capital por perna)
- Targets: **multi-horizonte** (1h, 4h, 8h, 24h; configuravel)
- Modelo principal MVP: **LightGBM** (fallback automatico para HistGradientBoosting se indisponivel)

## Proximos incrementos (v1)

1. Adicionar CatBoost como challenger oficial no script de treino.
2. Refinar custos por sessao/horario (spread dinamico).
3. Expandir features de congruencia (viradas de rank, persistencia de extremo).
4. Pipeline de inferencia diaria automatica (sinal em 21:00 BRT).

## Observacoes de fidelidade ao indicador

- `ignoreFuture=True` (nao-repaint) e o default.
- Formula de `dblPrev` preservada do MQ4: `(LWMA(shift+1)*231 + close(shift)*20) / 251`.
- Normalizacao por `ATR(100, shift+10)/10` preservada.
