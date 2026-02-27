# Decisoes Tecnicas (Pesquisa + Recomendacoes)

Data da pesquisa: **2026-02-27**.

## Decisoes confirmadas com o usuario

- Execucao: **Google Colab**.
- Persistencia: **Google Drive** como storage principal de dados/modelos/relatorios.
- Fonte de dados do MVP: **OANDA v20 API**.
- Basket return: **equal-dollar** (mesmo capital por perna; media simples das pernas).
- Modelagem: **LightGBM** como principal.
- Holding/target: **multi-horizonte configuravel** (nao fixar apenas 24h).

## 1) Arquitetura proposta (MVP -> v1)

### Modulos

- `forex_css.data`: ingestao, validacao de schema, timezone e persistencia.
- `forex_css.indicators`: port do MT4 (slope/CSS) e extensoes.
- `forex_css.features`: feature engineering multi-timeframe.
- `forex_css.dataset`: montagem do dataset no momento de decisao (21:00 BRT e variantes).
- `forex_css.models`: baseline + ML (walk-forward).
- `forex_css.backtest`: simulacao moeda->basket com custos.
- `scripts.*`: CLI fim-a-fim.

### Pipeline alvo

1. `download_data` -> `data/raw/{source}/{pair}/{tf}.parquet`
2. `build_features` -> `data/features/{tf}/...`
3. `build_dataset` -> `data/datasets/daily_21brt.parquet`
4. `train` -> `models/`
5. `backtest` -> `reports/`

## 2) Colab + Google Drive

### Pesquisa

- O Colab tem storage efemero no runtime; persistencia deve ficar em Drive/GCS ([FAQ Colab](https://research.google.com/colaboratory/faq.html)).
- Compatibilidade de runtime muda ao longo do tempo; Google recomenda pin de versoes quando necessario ([Runtime Version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)).
- Para Parquet no pandas, uso recomendado de `read_parquet/to_parquet` com engine `pyarrow` ([pandas read_parquet](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html), [pandas to_parquet](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)).

### Recomendacao

- **Colab-first com Drive como fonte de verdade** para `data/`, `models/`, `reports/`.
- Runtime do Colab apenas para computacao temporaria.
- Sempre salvar artefatos em caminho no Drive ao final de cada etapa.

### Trade-off

- Simplicidade vs performance: Drive e mais lento que disco local do runtime.
- Mitigacao: cache temporario em `/content` durante execucao e flush no fim.

## 3) Fontes de dados OHLC (historico + quase tempo real)

### Opcao A: Dukascopy (historico bulk)

- Fonte com historico longo via pagina oficial de historico ([Dukascopy Historical Data Feed](https://www.dukascopy.com/swiss/english/marketwatch/historical/)).
- Boa para baixar anos de dados e montar dataset offline.

Pros:
- Cobertura historica ampla.
- Bom para backtest longo.

Contras:
- Fluxo de download menos padronizado que APIs REST comerciais.
- Necessita padronizacao de timezone/schema no ingestion.

### Opcao B: OANDA v20 API

- Endpoint de candles oficial e granularidades definidas ([Instrument Candles endpoint](https://developer.oanda.com/rest-live-v20/instrument-ep/)).
- Guia oficial com limites e boas praticas de conexao/streaming ([Development Guide](https://developer.oanda.com/rest-live-v20/development-guide/)).

Pros:
- API consistente para producao (polling/stream).
- Integra bem com execucao real futura.

Contras:
- Historico pode ser mais limitado que datasets bulk dedicados.
- Dependencia de conta/ambiente da corretora.

### Opcao C: Twelve Data

- API de series temporais para forex ([Time Series docs](https://twelvedata.com/docs#/time-series)).
- Planos e limites por creditos/minuto ([Pricing](https://twelvedata.com/pricing)).

Pros:
- Integracao REST simples.
- Onboarding rapido para prototipo.

Contras:
- Limites/custos podem apertar para universos grandes multi-TF.
- Dependencia de plano para escala.

### Opcao D: Polygon Forex

- Endpoint de barras agregadas para forex ([Aggregates docs](https://polygon.io/docs/rest/forex/aggregates/custom-bars)).
- Planos com diferentes janelas historicas e acesso real-time ([Pricing](https://polygon.io/pricing)).

Pros:
- Documentacao robusta e boa DX.
- Boa opcao para dados near real-time.

Contras:
- Custo tende a subir para historico completo/latencia baixa.
- Requer revisar plano por classe de ativo antes de escalar.

### Recomendacao (MVP)

- **Fonte principal confirmada: OANDA v20**.
- Opcao de expansao futura: Dukascopy/HistData para bulk historico adicional, se necessario.

### Teste de amostra obrigatorio (antes de baixar anos)

- 1 par (`EURUSD`)
- 2 timeframes (`H1`, `D1`)
- 30 dias
- Validacoes:
  - schema/timezone
  - continuidade temporal
  - CSS sem explosoes numericas

## 4) Formato de armazenamento e schema

### Comparacao rapida

- Parquet: formato colunar, compressao eficiente, melhor para analytics ([Apache Parquet](https://parquet.apache.org/)).
- CSV: simples e universal, porem maior e mais lento.
- HDF5: bom desempenho, mas menos padrao no ecossistema atual de data lakes Python.

### Recomendacao

- **Parquet + pyarrow** como padrao.
- CSV apenas para ingestao inicial/intercambio.

### Schema padrao: candles

- `timestamp` (`datetime64[ns, UTC]`) - indice logico.
- `open`, `high`, `low`, `close` (`float64`)
- `volume` (`float64`, opcional)
- metadados no path: `source/pair/timeframe`

### Schema padrao: features (long)

- `timestamp`, `currency`, `timeframe`
- `css`, `css_prev`, `css_slope`, `css_sign`
- `rank_desc`, `rank_asc`
- `cross_zero_up`, `cross_zero_down`
- `cross_level_up`, `cross_level_down`
- `css_zscore`, `is_extreme`, `is_reversal`

## 5) Bibliotecas de ML (arvores)

### Opcoes pesquisadas

- scikit-learn (RandomForest / HistGradientBoosting): baseline robusto e simples ([RF docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [HGB docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)).
- LightGBM: eficiente, histogram-based, suporte a categorizacao e treino rapido ([LightGBM features](https://lightgbm.readthedocs.io/en/latest/Features.html)).
- CatBoost: bom com categoricas e estrategia para reduzir prediction shift ([CatBoost unbiased boosting](https://catboost.ai/docs/en/concepts/algorithm-main-stages_fighting-biases)).
- XGBoost: maduro, metodo `hist` muito eficiente ([XGBoost tree methods](https://xgboost.readthedocs.io/en/stable/treemethod.html)).

### Recomendacao (MVP)

- **MVP**: `RandomForest` (baseline) + `LightGBM` (modelo principal).
- **v1**: adicionar `CatBoost` para comparar robustez e calibracao.

## 6) Biblioteca de backtest

### Opcoes

- `vectorbt`: abordagem vetorizada e rapida, bom para explorar combinacoes de regras ([vectorbt docs](https://vectorbt.dev/)).
- `backtrader`: event-driven, boa modelagem de ordem/execucao ([backtrader docs](https://www.backtrader.com/docu/)).
- `zipline`/forks: historicamente relevante, mas ecossistema menos simples para setup atual.

### Recomendacao

- **MVP implementado**: engine proprio simplificado, focado em regra moeda->basket e custos fixos.
- **v1 (opcional)**: avaliar `vectorbt` (rapidez) ou `backtrader` (event-driven detalhado).

## 7) Regras para evitar leakage/lookahead

- Feature em `t` usa apenas candles fechados ate `t`.
- Targets comecam estritamente apos `t` (janela futura).
- Split temporal (walk-forward), sem embaralhar.
- Sincronizacao multi-TF por timestamp de decisao.

## 8) Trade-offs em aberto para v1

1. Custos no backtest:
   - MVP: spread + slippage fixos.
   - v1: curva de custo por sessao/horario (mais realista).
2. Modelo challenger:
   - MVP: LightGBM principal.
   - v1: adicionar CatBoost para comparativo robusto.
3. Engine de backtest:
   - MVP atual: engine proprio simplificado focado em moeda->basket.
   - v1: avaliar migracao para vectorbt/backtrader para maior extensibilidade.
