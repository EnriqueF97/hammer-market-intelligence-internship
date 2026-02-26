**Plan de Tesis**
# Modelado de Liquidez en WTI con IA: Asimetría y Temporalidad en el Impacto de Noticias
**Autor:** Enrique Favila Martínez\
Institución: Radboud University / Hammer Market Intelligence\
**Supervisora:** Dra. Lejla Batina\
**Duración estimada:** 5 meses\
**Commodity seleccionado:** WTI Crude Oil (CL)\
**Fuente de datos:** OilPriceAPI (intradía) + CME (Open Interest)

## 1. Contexto y Motivación
Este proyecto de investigación tiene como objetivo estudiar cómo las noticias afectan la liquidez en mercados de commodities, con un enfoque en:

1. **Asimetría:** ¿Las malas noticias afectan la liquidez de manera diferente a las buenas?

2. **Temporalidad:** ¿Cuánto tarda el mercado en reaccionar? ¿Cuánto persiste el efecto?

Se utiliza WTI (West Texas Intermediate) por su alta volatilidad a eventos geopolíticos, frecuencia de noticias, disponibilidad de datos intradía mediante OilPriceAPI, y abundante literatura de referencia.

El proyecto se enmarca en la intersección de:

- **Market Microstructure:** Estudio de la liquidez (bid-ask spread, profundidad, open interest).

- **Natural Language Processing (NLP):** Representación semántica de noticias.

- **Modelos temporales con IA:** Captura de efectos rezagados y no lineales.


## 2. Preguntas de Investigación
**RQ1 (Asimetría):**\
¿Tienen las noticias negativas (vs positivas) un impacto mayor y más persistente en las métricas de liquidez del WTI?

**RQ2 (Temporalidad):**\
¿Cuál es la estructura de retardo (lag structure) del impacto de las noticias en la liquidez? ¿Existen efectos de sob reacción y reversión?

**RQ3 (Extensión - si hay tiempo):**\
¿Cómo se propagan los shocks de liquidez entre commodities relacionados (ej. WTI, Brent, Gasolina)?

## 3. Variables del estudio
| Tipo | Variable | Descripción | Frecuencia | Fuente |
|---|---|---|---|---|
| Dependiente (Y) | Bid-Ask Spread | Diferencia entre mejor compra y mejor venta | Minuto a minuto | OilPriceAPI (derivado de precios) |
|  | Market Depth | Volumen en primeras posiciones del libro | Minuto a minuto | OilPriceAPI (si disponible) |
|  | Open Interest | Número de contratos abiertos | Diaria | CME (web scraping) |
| Independiente (X) | Representaciones de noticias | Embeddings (FinBERT), sentimiento léxico, keywords | Por evento | Hammer / APIs gratuitas |
| Control | Volatilidad intradía | Rango de precios en ventanas | Minuto a minuto | Calculado de OilPriceAPI |

## 4. Plan por Fases (5 Meses)
**Fase 1: Fundamentación y Diseño (2 semanas)**
- Revisión bibliográfica guiada (Tetlock, Smales, Diebold-Yilmaz, Borovkova).
- Refinamiento final de RQs.
- Familiarización con OilPriceAPI (plan gratuito).
- Definición de métricas exactas de liquidez según datos disponibles.

\
**Fase 2: Adquisición y Feature Engineering (4 semanas)**
- Implementar oilpriceapi_client.py para descargar datos intradía de WTI.
- Implementar cme_scraper.py para Open Interest diario.
- Implementar build_liquidity_features.py: cálculo de spread, profundidad, volumen acumulado.
- Implementar build_news_features.py con al menos:
    - Baseline: Loughran-McDonald lexicon.
    - Avanzado: FinBERT embeddings.
- Implementar build_event_windows.py para extraer ventanas [-30, +60] minutos alrededor de cada noticia.

**Fase 3: Modelado de Asimetría (RQ1) - 4 semanas**
- Modelo lineal con interacciones (regresión).
- Modelo de red neuronal con ramas separadas para sentimiento positivo/negativo.
- Evaluación comparativa.
- Visualización de resultados.

**Fase 4: Modelado Temporal (RQ2) - 5 semanas**
- Modelo VAR con funciones de impulso-respuesta.
- Modelo LSTM con atención.
- (Opcional) Transformer temporal.
- Análisis de persistencia y reversión.

**Fase 5: Extensión y Escritura (5 semanas)**
- (Si hay tiempo) RQ3: Spillovers con Graph Neural Networks.
- Escritura de tesis: Introducción, Related Work, Metodología, Resultados, Discusión.
- Preparación de dashboard simple para Hammer.


## 5. Técnicas de IA por fase
| Fase                  | Técnicas de IA                                                 | Propósito                                                    |
|-----------------------|----------------------------------------------------------------|--------------------------------------------------------------|
| Fase 2 (NLP)          | FinBERT, Sentence-BERT, embeddings contextuales                | Representación semántica de noticias                         |
| Fase 3 (Asimetría)    | Redes neuronales con ramas separadas, atención sobre polaridad | Capturar impacto diferencial de noticias positivas/negativas |
| Fase 4 (Temporalidad) | LSTM, GRU, Transformers, Neural Hawkes Processes               | Modelar evolución temporal de la liquidez post-noticia       |
| Fase 5 (Spillovers)   | Graph Neural Networks (GNNs)                                   | Modelar propagación entre commodities                        |
| Interpretabilidad     | Mecanismos de atención, SHAP, LIME                             | Explicar qué palabras y momentos importan                    |


## 6. Gestión de riesgos
| Riesgo                                   | Probabilidad | Impacto | Mitigación                                                                      |
|------------------------------------------|--------------|---------|---------------------------------------------------------------------------------|
| OilPriceAPI no tiene datos de bid-ask    | Alta         | Medio   | Usar volumen intradía y rango como proxy de liquidez                            |
| NLP muy lento/costoso                    | Media        | Medio   | Usar Sentence-BERT (más ligero) o embeddings pre-calculados                     |
| Modelos muy complejos para el tiempo     | Media        | Alto    | Priorizar modelos más simples pero bien ejecutados (VAR + LSTM)                 |
| No se obtienen resultados significativos | Baja         | Alto    | Tener baseline léxico como comparación; pivotar a análisis de eventos discretos |
| Tiempo insuficiente para RQ3             | Alta         | Bajo    | RQ3 es "nice to have"; priorizar RQ1 y RQ2                                      |


## 7. Entregables
1. **Documento de tesis** con:
    - Introducción y motivación.
    - Revisión de literatura.
    - Metodología detallada (reproducible).
    - Resultados de RQ1 y RQ2.
    - Discusión y conclusiones.
2. **Repositorio de código** (GitHub) con la arquitectura modular documentada.
3. **Dashboard básico** para Hammer (opcional, si los recursos y tiempo se alinean).

## 8. Cronograma visual
```
Semana: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
Fase 1:  [X X]
Fase 2:    [X X X X]
Fase 3:          [X X X X]
Fase 4:                [X X X X X]
Fase 5:                         [X X X X X]
```

## 9.  Arquitectura modular del proyecto
Se diseña una arquitectura que permite escalar de datos diarios a intradía sin rehacer el proyecto.

```
proyecto_wti/
│
├── 01_data/
│   ├── raw/                  # Datos sin procesar (CSV descargados)
│   ├── processed/            # Datos limpios y alineados
│   └── features/              # Features listos para modelar
│
├── 02_notebooks/              # Exploración y prototipado
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_nlp_prototyping.ipynb
│   └── 03_modelos_iniciales.ipynb
│
├── 03_src/                     # Código fuente modular
│   ├── adquisicion/
│   │   ├── oilpriceapi_client.py    # Cliente para API
│   │   ├── cme_scraper.py           # Scraping de Open Interest
│   │   └── eia_downloader.py        # Descarga de inventarios
│   │
│   ├── features/
│   │   ├── build_liquidity_features.py   # Calcula spread, depth, etc.
│   │   ├── build_news_features.py        # NLP invariante (FinBERT, etc.)
│   │   └── build_event_windows.py        # Ventanas alrededor de noticias
│   │
│   ├── modelos/
│   │   ├── asymmetry/                    # Modelos para RQ1
│   │   │   ├── linear_interaction.py
│   │   │   └── neural_asymmetry.py
│   │   ├── temporal/                      # Modelos para RQ2
│   │   │   ├── var_model.py
│   │   │   ├── lstm_model.py
│   │   │   └── transformer_model.py
│   │   └── evaluation/                     # Métricas y validación
│   │       ├── metrics.py
│   │       └── plots.py
│   │
│   └── config/                             # Configuración por experimento
│       ├── config_daily.yaml
│       └── config_intraday.yaml
│
├── 04_outputs/                 # Resultados, gráficos, tablas
│   ├── figures/
│   ├── tables/
│   └── models/                  # Modelos entrenados guardados
│
├── 05_informes/                 # Avances y entregables
│   ├── revision_bibliografica.md
│   ├── informe_fase1.md
│   └── ...
│
├── requirements.txt             # Dependencias
└── README.md                    # Descripción del proyecto
````

