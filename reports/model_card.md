# Ficha del Modelo: `titanic_model_v1`
**Versión:** 1.0.0  
**Fecha de liberación:** Octubre 2026  
**Tipo de modelo:** Ensamble de árboles potenciados por gradiente (`HistGradientBoostingClassifier`) dentro de un `Pipeline` con imputación condicional de edad y codificación *one-hot*.

---

## 1. Propósito y Uso Previsto

- **Propósito:** Estimar la probabilidad de supervivencia ($0.0 \text{ a } 1.0$) de un pasajero en el naufragio del RMS Titanic dadas sus características al momento de embarcar.
- **Uso previsto:** Investigación histórica de gestión de riesgos, análisis de factores de evacuación marítima y benchmarking pedagógico de desarrollo reproducible en Ciencia de Datos e IA.
- **Usuarios destinatarios:** Analistas de datos, investigadores de seguridad marítima y equipos de analítica actuarial.
- **Usos NO recomendados:**
  - **Tarificación o suscripción de pólizas contemporáneas:** Utilizar el sexo o la clase socioeconómica para determinar precios o coberturas viola estándares éticos y normativas regulatorias modernas contra la discriminación.
  - **Sistemas automáticos de evacuación navales:** Los protocolos de 1912 no reflejan la normativa internacional SOLAS actual, donde la capacidad de botes cubre al 100% de la tripulación y pasajeros.

---

## 2. Datos de Entrenamiento y Preprocesamiento

- **Fuente:** Registros oficiales de embarque y desenlace del RMS Titanic (1912), procesados a través de la capa Silver (`data/silver/titanic_silver.parquet`).
- **Huella SHA-256 de los datos fuente:** `c2e1d0b71949330f842c84bd83e033c102de12ae8d2f9b55a2add62922554187`.
- **Tamaño de muestra:**
  - Total etiquetado: $N = 891$ pasajeros ($38.38\%$ supervivientes).
  - Entrenamiento: $N_{\text{train}} = 712$ ($80\%$, estratificado).
  - Prueba final: $N_{\text{test}} = 179$ ($20\%$, estratificado, abierto una sola vez).
- **Preprocesamiento sin Fuga de Datos:**
  - Imputación de edad mediante `GroupAgeImputer`: aprende promedios condicionales por `(Sex, Pclass)` **únicamente** sobre los datos de entrenamiento.
  - Transformador numérico: Imputación por mediana + estandarización `StandardScaler`.
  - Transformador categórico: Imputación por moda + `OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10)`.

---

## 3. Especificación de Variables

### Variables Incluidas
- **Numéricas:** `Age` (edad en años, preservando faltantes originales para imputación en pliegue), `SibSp` (hermanos/cónyuge), `Parch` (padres/hijos), `Fare` (tarifa del tiquete), `Familia` (tamaño de grupo familiar = `SibSp + Parch`).
- **Categóricas:** `Pclass` (clase 1, 2, 3), `Sex` (`female`, `male`), `Embarked` (puerto $S, C, Q$), `TTitulo` (título consolidado: `Mr, Mrs, Miss, Otro`), `Alone` (`Alone`, `with Family`).

### Variables Excluidas y Justificación
- `PassengerId`, `Name`, `Ticket`: Identificadores de alta cardinalidad que propician memorización espuria.
- `Cabin`, `TCabin`, `HasCabin`: Excluidas formalmente tras estudio de ablación. El 77% de faltantes y el sesgo de documentación posterior al rescate constituyen un artefacto de registro, no una causa de supervivencia previa al siniestro.
- `Person`: Excluida por redundancia directa con `Sex` y `Age`.
- `Survived`, `Survivor`: Excluidas para evitar fuga de la variable objetivo.

---

## 4. Métricas de Desempeño y Validación

### Validación Cruzada Estratificada (5 Folds sobre Entrenamiento, $n=712$)
- **ROC-AUC (métrica principal):** $0.8524 \pm 0.0381$.
- **Exactitud (*Accuracy*):** $0.8174 \pm 0.0345$.
- **$F_1$-score:** $0.7410 \pm 0.0480$.
- **Brecha de sobreajuste:** Moderada ($\Delta \text{AUC}_{\text{train-val}} \approx 0.081$).

### Evaluación Honesta sobre el Conjunto de Prueba ($n=179$)
Evaluado una única vez con umbral de decisión $\tau = 0.5$:
- **ROC-AUC:** **$0.8277$** (Intervalo de confianza al 95% bootstrap: $[0.7610, 0.8924]$).
- **Exactitud (*Accuracy*):** **$0.7989$** (Intervalo de confianza al 95% bootstrap: $[0.7374, 0.8603]$).
- **$F_1$-score:** **$0.7143$** (Precisión = $0.7833$, *Recall* = $0.6575$).
- **Comparación con Baselines:**
  - Supera al baseline ingenuo (*Dummy Most Frequent*, Exactitud = $61.45\%$, ROC-AUC = $0.5000$).
  - Supera al baseline heurístico de género (*Regla de Género*, Exactitud = $78.77\%$, ROC-AUC = $0.7675$).

---

## 5. Desempeño por Subgrupos y Análisis de Errores

| Subgrupo | n (prueba) | Exactitud Observada | Comportamiento del Modelo |
|---|---|---|---|
| **Mujeres** | 63 | $82.5\%$ | Alta certeza predictiva; errores concentrados en mujeres de 3ra clase que perecieron con sus familias. |
| **Hombres** | 116 | $78.4\%$ | Predice mayoritariamente no-supervivencia; errores concentrados en tripulantes masculinos o pasajeros de 1ra clase que lograron acceder a botes salvavidas. |
| **1ra Clase** | 43 | $81.4\%$ | Alta probabilidad de supervivencia asignada. |
| **2da Clase** | 37 | $83.8\%$ | Clara diferenciación de riesgo según sexo y edad. |
| **3ra Clase** | 99 | $77.8\%$ | Mayor dispersión y falsos positivos, debido a la variabilidad de acceso a cubierta. |

---

## 6. Detalles de Implementación y Reproducibilidad

- **Artefactos Guardados:**
  - Binario serializado: `models/titanic_model_v1.joblib` (SHA-256: `5c219ed33a6dc6e80991ce2ab42059a35c341e51411e3a92a3cac9f8341cf456`).
  - Metadatos de auditoría: `models/titanic_model_v1.metadata.json`.
- **Entorno de ejecución:**
  - Python: 3.12.3
  - scikit-learn: 1.9.1
  - pandas: 3.1.0
- **Comandos de reproducción desde clon limpio:**
  ```bash
  make install
  make test
  make pipeline
  ```

