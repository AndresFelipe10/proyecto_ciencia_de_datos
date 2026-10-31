# Buenas Prácticas en Ciencia de Datos e IA

## Propósito del taller

Realizar un **análisis exploratorio de datos (EDA)** completo sobre el dataset del Titanic y, a partir de él, **diseñar, evaluar y guardar un modelo de machine learning** que prediga la supervivencia de un pasajero. Todo se construye como un proyecto mantenible: reproducible, verificable, documentado y preparado para crecer. Se utilizan Python, pandas, NumPy, matplotlib, seaborn, SciPy y scikit-learn. PyTorch no se usa: en datos tabulares pequeños como este, los modelos clásicos son suficientes y más fáciles de explicar; las redes neuronales se incorporan solo cuando el problema las requiere.

> **Resultado esperado**: un proyecto estructurado y organizado siguiendo las mejores prácticas de desarrollo para analítica e IA, que se clonar, instalar, ejecutar (`make pipeline`), probar (`make test`) y entender sin depender de rutas locales ni de pasos manuales ocultos, y que termine produciendo un modelo guardado con sus metadatos.

### Alcance

Al terminar serás capaz de:

1. Organizar un proyecto de datos por responsabilidades (ingesta, ingeniería de datos, agregación, modelado) con la arquitectura **medallion** (Bronze, Silver, Gold).
2. Inspeccionar un dataset nuevo y **diagnosticar datos faltantes** con tablas, mapas de calor y un reporte de calidad automático.
3. Construir **variables nuevas** (*feature engineering*) como **funciones puras con pruebas**: `Person`, `Alone`, `TCabin`, `TTitulo`.
4. Elegir y construir el gráfico adecuado con Seaborn y **justificar** estrategias de imputación reconociendo el sesgo que introducen.
5. Distinguir entre **descripción**, **asociación** y **causalidad**, y comunicar con honestidad las limitaciones de tu análisis.
6. Formular un problema de clasificación, definir métricas y **baselines**, y evitar la **fuga de datos** (*data leakage*) con `Pipeline` y `ColumnTransformer`.
7. Comparar modelos con **validación cruzada**, ajustar hiperparámetros, analizar errores e interpretar el modelo.
8. Evaluar **una sola vez** sobre un conjunto de prueba, entrenar el modelo final y **guardarlo** con metadatos que permitan auditarlo.

## 1. Contexto del problema

Imagina que eres analista de datos en una **empresa de seguros** que investiga patrones históricos de supervivencia en desastres marítimos. La gerencia quiere saber, a partir del caso del Titanic (1912), **qué características de una persona aumentaron o disminuyeron su probabilidad de sobrevivir**, para alimentar modelos de riesgo y protocolos de evacuación.

> **Pregunta Central (EDA)**: ¿Qué factores tuvieron más influencia sobre la supervivencia de los pasajeros del Titanic?
> **Pregunta Predictiva (ML)**: Dadas las características de un pasajero conocidas al embarcar, ¿con qué probabilidad habría sobrevivido, y qué tan confiable es esa estimación?

**¿Por qué EDA antes del modelo?** Porque antes de predecir hay que *entender*. El análisis exploratorio revela errores y vacíos en los datos, genera hipótesis y evita construir modelos sobre supuestos falsos. En este dataset **faltan muchos datos** (edad, cabina) y la forma de rellenarlos **cambia las conclusiones**.

### Diccionario de datos

| Variable | Descripción | Tipo esperado |
|---|---|---|
| `PassengerId` | Identificador del pasajero | entero |
| `Survived` | 0 = no sobrevivió, 1 = sobrevivió | entero (categórica binaria) |
| `Pclass` | Clase del tiquete: 1 = alta, 2 = media, 3 = baja | entero (categórica ordinal) |
| `Name` | Nombre completo (incluye el título: Mr, Mrs, Miss…) | texto |
| `Sex` | `male` / `female` | categórica |
| `Age` | Edad en años | decimal (con faltantes) |
| `SibSp` | Nº de hermanos/cónyuge a bordo | entero |
| `Parch` | Nº de padres/hijos a bordo | entero |
| `Ticket` | Número de tiquete | texto |
| `Fare` | Tarifa pagada | decimal |
| `Cabin` | Cabina (la letra indica la cubierta) | texto (muchos faltantes) |
| `Embarked` | Puerto de embarque: C = *Cherbourg*, Q = *Queenstown*, S = *Southampton* | categórica |

> **Nota sobre los datos.** El dataset contiene pasajeros de quienes **sabemos** si sobrevivieron (primeras 891 filas, "etiquetados") y pasajeros de quienes **no** (el resto, "sin etiqueta"). En este taller solo se modela con el primer grupo; el segundo se conserva en Bronze y se usa en el Taller #3.

## 2. Principios de diseño

### 2.1 Organizar el código por responsabilidad

Cada módulo tiene una responsabilidad clara. La ingesta obtiene los datos, la ingeniería de datos los valida y transforma, la agregación produce tablas de consumo y el módulo de modelos entrena y evalúa. `job.py` coordina esas piezas, pero no contiene la lógica. Las transformaciones se escriben como **funciones puras** (reciben un `DataFrame`, devuelven uno nuevo, no modifican la entrada).

**Por qué es buena práctica**
- Facilita localizar y corregir errores.
- Permite probar cada transformación sin ejecutar el flujo completo.
- Reduce duplicación y acoplamiento entre módulos.
- Hace posible reutilizar una transformación en más de un flujo.

### 2.2 Separar código, datos, configuración y documentación

El código fuente se versiona; los datos generados y los modelos (archivos pesados o derivados) no. La configuración no sensible (URL, semilla, tamaño del conjunto de prueba) se mantiene fuera de la lógica, en `.env` y `config.py`. Las decisiones y contratos se documentan junto al flujo al que pertenecen.

**Por qué es buena práctica**
- El repositorio no crece con datos derivados ni binarios.
- Varias personas pueden trabajar con el mismo código.
- Los parámetros que afectan la ejecución son explícitos.
- Se puede auditar qué se esperaba recibir y qué produce cada flujo.

### 2.3 Aplicar la arquitectura medallion con criterio

Usamos tres capas locales:

- **Bronze**: copia de los datos tal como llegan (CSV), con modificaciones mínimas y un archivo de metadatos de ingesta (fuente, fecha, huella SHA-256).
- **Silver**: datos validados, tipados, normalizados y enriquecidos (Parquet). Aquí viven las variables derivadas y es la entrada tanto del EDA como del modelado.
- **Gold**: tablas agregadas orientadas a un consumo concreto: el informe de hallazgos.

Las capas describen el estado y el propósito de los datos, no necesariamente tecnologías distintas. En un entorno real, la retención, el formato y los controles de acceso se definen según el proyecto.

### 2.4 Diseñar para reproducibilidad

Una ejecución debe poder repetirse con las mismas entradas y parámetros: fuentes documentadas, dependencias fijadas, semillas aleatorias en la configuración y cero rutas absolutas. Los modelos guardados incluyen versiones de librerías y huella de los datos de entrenamiento.

**Por qué es buena práctica**
- Los resultados se comparan entre ejecuciones y personas.
- Los errores se reproducen.
- Las evaluaciones de modelos son más confiables.

### 2.5 Verificar con pruebas automáticas

Cada módulo trae pruebas en `tests/`. Una prueba describe en código lo que la función **promete** (por ejemplo: "un pasajero de 2 años es `child`"). En este taller las pruebas ya están escritas: tu trabajo es implementar las funciones hasta que pasen (`make test`).

**Por qué es buena práctica**
- Detectas regresiones al cambiar una regla de limpieza.
- Las pruebas documentan decisiones (¿qué pasa con un pasajero sin edad?).
- Permiten refactorizar con confianza.

### 2.6 Evaluar sin fuga de datos

La fuga de datos ocurre cuando información del conjunto de evaluación influye en el entrenamiento (por ejemplo, imputar la edad con el promedio de **todos** los pasajeros antes de dividir). Se evita ajustando todo el preprocesamiento **solo con datos de entrenamiento**, dentro de un `Pipeline`, y abriendo el conjunto de prueba **una única vez**.

**Por qué es buena práctica**
- Las métricas reportadas son estimaciones honestas del desempeño real.
- Evita decisiones basadas en el conjunto de prueba (que dejaría de ser "prueba").

