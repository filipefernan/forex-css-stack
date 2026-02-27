# CurrencySlopeStrength.mq4 - resumo de port

Arquivo analisado: `currency-slope-strength.mq4` (raiz do projeto).

## Escopo do port

Portar apenas a logica de calculo do indicador (sem UI/objetos/alertas):

1. `getSlope(symbol, tf, shift)`
2. `calcCSS(tf, shift)`
3. `calcTma(symbol, tf, shift)` apenas para modo `ignoreFuture=false` (opcional no MVP)
4. Regras de normalizacao e agregacao por moeda (8 moedas)

## Parametros relevantes no MQ4

- `symbolsToWeigh`: lista de pares a ponderar.
- `ignoreFuture` (default `true`): caminho nao-repaint.
- `addSundayToMonday` (default `true`): ajuste especial em `D1` quando ha candle de domingo.
- `levelCrossValue` (default `0.20`): referencia para features de cruzamento.
- `maxBars`: limite de calculo no grafico (nao necessario no backend Python batch).

## Formulas centrais

### `getSlope` (modo `ignoreFuture=true`, default)

Para um par e barra `shift`:

- `atr = ATR(100, shift + 10) / 10`
- `dblTma = LWMA(close, 21, shift)`
- `dblPrev = (LWMA(close, 21, shift+1) * 231 + close(shift) * 20) / 251`
- `slope = (dblTma - dblPrev) / atr`

Observacoes:
- O uso de `shift+10` no ATR esta no codigo original e foi preservado.
- `dblPrev` usa os coeficientes `231` e `20` do codigo fonte original.

### `getSlope` (modo `ignoreFuture=false`)

- `dblTma = calcTma(shift)` (usa barras futuras e passadas)
- `dblPrev = calcTma(shift+1)`
- `slope = (dblTma - dblPrev) / atr`

### `calcCSS`

Para cada simbolo:
- soma `slope` na moeda base
- subtrai `slope` na moeda cotada

Depois:
- divide cada moeda pelo numero de ocorrencias da moeda na cesta de simbolos.

Resultado:
- 8 series CSS (USD, EUR, GBP, CHF, JPY, AUD, CAD, NZD).

## Ajuste de domingo (D1)

Se `addSundayToMonday=true` e existirem candles de domingo:
- quando a barra atual em `D1` cai no domingo, o codigo incrementa `shift` em 1 (`shiftWithoutSunday++`) para os calculos de slope.

No port Python, isso foi implementado como ajuste equivalente da serie no domingo.

## O que foi explicitamente ignorado no port

- Tabela visual, labels, objetos graficos.
- Alertas (`Popup`, `Email`, `Push`), bullets, cruzes coloridas.
- Layout/estilo da janela do indicador no MT4.
