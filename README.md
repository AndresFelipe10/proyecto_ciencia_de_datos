# Predicción de Supervivencia en el Titanic

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn 1.9.1](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![Pipeline](https://img.shields.io/badge/pipeline-Medallion-6F42C1)](#pipeline-de-datos-e-ingeniería-de-características)

Proyecto de Ciencia de Datos y MLOps para estimar la probabilidad de supervivencia de pasajeros del RMS Titanic a partir de la información disponible al momento de embarcar. El problema se formula como una clasificación binaria supervisada con `Survived` como variable objetivo.

El repositorio prioriza un diseño modular y reproducible: arquitectura de datos Medallion (`bronze`/`silver`/`gold`), pipeline de preprocesamiento sin fuga de datos, pruebas automatizadas, notebooks de análisis y versionado del artefacto y la metadata del modelo.

> **Alcance y uso responsable:** el modelo es un artefacto histórico y didáctico. No debe utilizarse para evacuación moderna, tarificación o suscripción de seguros, ni para decisiones basadas en sexo o clase socioeconómica.

## Estructura del repositorio

```text
.
├── data/
│   ├── bronze/       # Datos ingeridos y reportes de calidad
│   ├── silver/      # Datos limpios y transformados en Parquet
│   └── gold/        # Agregaciones para análisis y reportes
├── flows/            # Ingestión, ingeniería, agregación y entrenamiento
│   ├── 00_ingestion.py
│   ├── 01_data_engineering.py
│   ├── 03_aggregation.py
│   ├── 05_ml_models.py
│   └── job.py        # Orquestador por etapas
├── models/           # Modelo serializado y metadata de auditoría
├── notebooks/        # Exploración y modelado reproducible
├── reports/          # Problema ML, conclusiones EDA y model card
├── tests/            # Pruebas unitarias e integración del pipeline
├── Makefile          # Comandos operativos
├── pyproject.toml    # Dependencias y configuración del proyecto
└── .env.example      # Configuración local reproducible
```

## Pipeline de datos e ingeniería de características

El flujo ejecuta las etapas en orden:

1. **Bronze:** descarga o lee los datos etiquetados y no etiquetados y conserva metadatos de ingestión y calidad.
2. **Silver:** valida, limpia y transforma los datos a Parquet (`titanic_silver.parquet` y `titanic_final.parquet`).
3. **Gold:** genera tablas agregadas para el EDA, como supervivencia por sexo, clase, puerto, familia y condición de viaje solo/acompañado.
4. **Train:** construye el pipeline de scikit-learn, realiza la evaluación y guarda el modelo y su metadata.

El EDA (`reports/conclusiones_eda.md`) analiza 891 pasajeros etiquetados y una supervivencia global de 38.38%. Las asociaciones históricas más fuertes se observan en `Sex` y `Pclass`, con interacciones relevantes entre sexo, clase y edad. También documenta la confusión entre `Embarked` y clase, el efecto proxy de `Fare` y el sesgo producido por valores faltantes de edad.

### Variables derivadas

- `Familia`: número de acompañantes, calculado como `SibSp + Parch`.
- `Alone`: `Alone` cuando `Familia == 0`; `with Family` en caso contrario.
- `TTitulo`: título extraído de `Name` y consolidado en `Mr`, `Mrs`, `Miss` u `Otro`.
- `AgeOriginal`: conserva la edad faltante original para evitar imputación previa al entrenamiento.
- `TCabin` y `HasCabin`: se generan durante ingeniería y EDA, pero se excluyen del modelo final por su riesgo de artefacto retrospectivo.

### Variables utilizadas por el modelo

| Tipo | Variables |
|---|---|
| Numéricas | `Age`, `SibSp`, `Parch`, `Fare`, `Familia` |
| Categóricas | `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone` |
| Objetivo | `Survived` |

El modelo usa `AgeOriginal` como entrada de edad y aplica `GroupAgeImputer`, que aprende la media de edad por `(Sex, Pclass)` únicamente con el conjunto de entrenamiento. Después se aplica imputación numérica, `StandardScaler`, imputación categórica y `OneHotEncoder` con manejo de categorías infrecuentes. Los identificadores (`PassengerId`, `Name`, `Ticket`), la variable objetivo y las variables de cabina se excluyen para evitar memorización o fuga de información.

## Modelado y rendimiento

El estimador seleccionado es `HistGradientBoostingClassifier`, registrado como `hgb`, dentro de un `Pipeline` de scikit-learn. La métrica principal es ROC-AUC; el umbral de decisión para las métricas de clasificación es `0.5`.

### Hiperparámetros

| Parámetro | Valor |
|---|---:|
| `learning_rate` | `0.05` |
| `max_depth` | `3` |
| `max_iter` | `200` |
| `l2_regularization` | `1.0` |
| `random_state` | `42` |

### Evaluación final

La metadata de `titanic_model_v1` reporta una partición estratificada de 80/20: 712 observaciones para entrenamiento y 179 para test. El conjunto de test se evaluó una sola vez.

| Métrica | Test |
|---|---:|
| Accuracy | **0.7989** |
| F1-Score | **0.7143** |
| ROC-AUC | **0.8277** |
| Tamaño del test | **179** |
| Umbral | **0.5** |

Como referencia, la validación cruzada estratificada de cinco folds sobre entrenamiento obtuvo ROC-AUC `0.8524 ± 0.0381`, accuracy `0.8174 ± 0.0345` y F1 `0.7410 ± 0.0480`. El modelo supera los baselines documentados de clase mayoritaria y regla de género, aunque la incertidumbre es relevante por el tamaño reducido del conjunto de prueba.

## Instalación y configuración

### Requisitos

- Python **3.12 o superior** (el artefacto fue generado con Python 3.12.3).
- `git` y `make`.
- Archivo `.env` local, creado a partir de `.env.example`.

```bash
git clone https://github.com/AndresFelipe10/proyecto_ciencia_de_datos.git
cd proyecto_ciencia_de_datos

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cp .env.example .env
pip install -e ".[dev]"
```

La configuración por defecto define `TITANIC_URL`, `RANDOM_STATE=42`, `TEST_SIZE=0.2`, `CV_FOLDS=5` y `TITANIC_N_LABELED=891`. El archivo `.env` no debe versionarse.

## Ejecución y testing

Desde la raíz del repositorio y con el entorno virtual activo:

```bash
# Ejecutar todo el pipeline: bronze -> silver -> gold -> entrenamiento
make pipeline

# Ejecutar etapas individuales
make ingest
make silver
make gold
make train

# Ejecutar pruebas unitarias e integración
pytest tests/
# Equivalente mediante Makefile
make test

# Validar estilo
make lint
```

El orquestador también permite ejecutar el flujo directamente:

```bash
python -m flows.job --stage all
```

Los artefactos generados se guardan en `data/`, `models/titanic_model_v1.joblib` y `models/titanic_model_v1.metadata.json`. La metadata registra versiones de Python, pandas y scikit-learn, hiperparámetros, métricas, semillas y hashes SHA-256 de los datos y del modelo para facilitar auditoría y reproducibilidad.

## Documentación adicional

- [`reports/problema_ml.md`](reports/problema_ml.md): formulación, baselines, leakage, ablación y limitaciones.
- [`reports/conclusiones_eda.md`](reports/conclusiones_eda.md): hallazgos y sesgos del análisis exploratorio.
- [`reports/model_card.md`](reports/model_card.md): propósito, variables, métricas y uso previsto del modelo.
- [`notebooks/02_exploracion.ipynb`](notebooks/02_exploracion.ipynb): exploración y visualizaciones.
- [`notebooks/04_modelado.ipynb`](notebooks/04_modelado.ipynb): comparación, selección y evaluación de modelos.