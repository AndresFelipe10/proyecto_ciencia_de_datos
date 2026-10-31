# Formulación del Problema y Auditoría Metodológica de ML
**Proyecto:** Predicción de Supervivencia en Desastres Marítimos (Titanic)  
**Dominio:** Analítica de Riesgos y Modelado Actuarial de Seguros  
**Fecha:** Octubre 2026

---

## 1. Definición del Problema de Negocio y Métricas (Paso 39)

| Decisión Metodológica | Especificación y Respuesta | Justificación Técnica y de Negocio |
|---|---|---|
| **Tipo de problema** | Clasificación binaria supervisada | La variable a predecir tiene dos estados exhaustivos y excluyentes: sobrevivió ($y=1$) o no sobrevivió ($y=0$). |
| **Variable objetivo** | `Survived` | Variable binaria observable tras el siniestro. |
| **Unidad de análisis y momento de predicción** | Un pasajero, **al momento de embarcar** | Las características predictivas deben conocerse en el puerto antes de que el buque zarpe (edad, sexo, clase, acompañantes, tarifa). Cualquier dato posterior introduce fuga de datos temporal. |
| **Métrica principal** | **ROC-AUC** | Mide la capacidad del modelo de ordenar a los pasajeros según su probabilidad de riesgo independientemente de un umbral arbitrario de corte. Es robusta ante la distribución de clases ($\approx 38\%$ positivos). |
| **Métricas secundarias** | Exactitud (*Accuracy*), $F_1$-score, Precisión, *Recall* | Permiten evaluar el desempeño operativo del clasificador una vez fijado el umbral de decisión ($\tau = 0.5$). |
| **Costo asimétrico de errores** | Falso Positivo (predecir sobrevida a quien perece) vs. Falso Negativo (predecir muerte a quien sobrevive) | Para una aseguradora marítima, un **falso positivo** implica subestimar la probabilidad de fallecimiento del asegurado, llevando a una reserva de capital deficiente y tarificación insuficiente de la prima. Un **falso negativo** sobreestima el riesgo, provocando primas poco competitivas. |
| **Criterio de éxito** | Superar el baseline heurístico de género ($\approx 0.787$ accuracy, $\approx 0.81$ ROC-AUC) | Un modelo de machine learning complejo solo se justifica si supera a una regla simple humana de evacuación. |

---

## 2. Auditoría de Variables y Prevención de Fuga de Datos (Paso 40)

| Variable en Silver | ¿Existe antes del evento? | ¿Riesgo de fuga (*leakage*)? | Decisión y Tratamiento |
|---|---|---|---|
| `Survived`, `Survivor` | No | **Fuga directa** (variable objetivo) | **Excluir** de la matriz de entrenamiento $X$. |
| `PassengerId`, `Name`, `Ticket` | Sí | Memorización espuria e identificadores únicos | **Excluir**. Se extrae únicamente la señal sociodemográfica del título mediante `TTitulo`. |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | — | Orden ordinal impuesto arbitrariamente sobre variables nominales | **Excluir**. Se utiliza codificación *One-Hot* dentro del `Pipeline` para evitar falsas relaciones lineales de orden. |
| `Age` (imputada en Silver) | No | **Fuga estadística** (la media se calculó con todo el dataset, incluyendo prueba) | **Excluir**. Se carga `AgeOriginal` con sus valores nulos originales para que el modelo aprenda la imputación exclusivamente en $X_{\text{train}}$. |
| `Cabin`, `TCabin`, `HasCabin` | **Dudoso** | **Artefacto de registro post-evento** (los sobrevivientes tuvieron mayor probabilidad de registrar o recordar su cabina) | **Excluir del modelo final** tras estudio de ablación para evitar explotar un sesgo de captura histórica. |
| `Person` | Derivada | Redundante con `Sex` y `Age` | **Excluir** del modelo predictivo base para evitar multicolinealidad. |

---

## 3. Baselines de Referencia (Paso 42)

Evaluados mediante validación cruzada estratificada de 5 pliegues sobre $X_{\text{train}}$ ($n = 712$):

1. **Baseline Ingenuo (*Dummy Most Frequent*):**
   - Siempre predice la clase mayoritaria ($0 = \text{no sobrevive}$).
   - **Exactitud:** $\approx 61.66\%$.
   - **ROC-AUC:** $0.5000$.
   - **$F_1$-score:** $0.0000$ (no detecta ningún sobreviviente).

2. **Baseline Heurístico (*Regla de Género*):**
   - Asigna $1$ si `Sex == 'female'` y $0$ si `Sex == 'male'`.
   - **Exactitud:** $\approx 78.65\%$.
   - **ROC-AUC:** $\approx 0.7675$.
   - **$F_1$-score:** $\approx 0.7161$.
   - *Este es el estándar mínimo a batir por cualquier modelo estadístico.*

---

## 4. Selección de Modelos Finalistas (Paso 50)

Tras comparar 6 familias de algoritmos bajo el mismo preprocesamiento y pliegues estratificados:

1. **HistGradientBoostingClassifier (`hgb`):**
   - **ROC-AUC:** $0.8524 \pm 0.0381$.
   - **Exactitud:** $0.8174 \pm 0.0345$.
   - **Brecha de sobreajuste:** Moderada ($\approx 0.08$). Maneja interacciones no lineales complejas entre edad, tarifa y clase sin requerir normalización previa. **Seleccionado como estimador principal.**
2. **LogisticRegression (`logreg`):**
   - **ROC-AUC:** $0.8415 \pm 0.0402$.
   - **Exactitud:** $0.7963 \pm 0.0267$.
   - **Brecha de sobreajuste:** Muy baja ($\approx 0.015$). Altamente interpretable mediante *odds ratios*. **Seleccionado como modelo de referencia explicativo.**
3. **RandomForestClassifier (`forest`):**
   - Presenta una brecha de sobreajuste excesiva ($\approx 0.14$), memorizando el conjunto de entrenamiento sin superar a `hgb` en validación.

---

## 5. Estudio de Ablación y Decisión sobre Cabina (Paso 61)

Se comparó el desempeño de validación cruzada bajo cuatro configuraciones:
- **A) Base** (Numéricas + Categóricas nominales sin cabina): ROC-AUC $\approx 0.852$.
- **B) Base + Cabina** (`HasCabin`, `TCabin`): ROC-AUC $\approx 0.854$ ($\Delta \text{AUC} \approx +0.002$).
- **C) Sin Título**: ROC-AUC $\approx 0.844$ ($\Delta \text{AUC} \approx -0.008$).
- **D) Sin Sexo**: ROC-AUC $\approx 0.748$ ($\Delta \text{AUC} \approx -0.104$).

> **Decisión:** Aunque incluir cabina muestra un incremento marginal de $+0.002$ en ROC-AUC, **se excluye formalmente del modelo definitivo (`USE_CABIN_FEATURES = False`)**. El 77% de faltantes y el hecho documentado de que los sobrevivientes proporcionaron testimonios detallados de sus camarotes en comisiones de investigación confirma que la presencia de cabina es un artefacto retrospectivo de captura, no un factor predictivo causante de evacuación.

---

## 6. Consideraciones Éticas y Limitaciones Operativas (Pasos 54 y 63)

1. **Variables Sensibles y Discriminación:**
   - El modelo otorga el mayor peso predictivo a `Sex` y `Pclass`.
   - Si bien esto describe fielmente la evacuación de 1912, utilizar el género o la clase socioeconómica para determinar coberturas, primas o indemnizaciones en pólizas modernas infringe normativas internacionales contra la discriminación actuarial y derechos humanos.
2. **Incertidumbre Muestral:**
   - Con un conjunto de prueba de solo 179 pasajeros, el intervalo de confianza al 95% por bootstrap sobre ROC-AUC ronda $[0.76, 0.89]$. Cualquier comparación de modelos con diferencias menores a 2 puntos porcentuales no es estadísticamente concluyente.
3. **Ámbito de Aplicación:**
   - El modelo es un artefacto analítico estrictamente histórico y didáctico. **No debe utilizarse** en sistemas de control de evacuación modernos ni en decisiones operativas navales contemporáneas.

