# Informe Ejecutivo: Análisis Exploratorio de Datos del Titanic
**Destinatario:** Gerencia de Riesgos y Modelado de Seguros  
**Fecha:** Octubre 2026  
**Objetivo:** Identificar patrones históricos de supervivencia y evaluar la viabilidad de utilizar los datos del desastre del Titanic (1912) en protocolos de riesgo marítimo.

---

## 1. Hallazgos Principales del Análisis

A partir del análisis sobre los 891 pasajeros con desenlace registrado ($n = 891$), la tasa general de supervivencia fue del **38.38%** ($n = 342$). Los factores con mayor asociación estadística con la supervivencia fueron:

1. **Protocolo de evacuación y prioridad por género:**
   - Las mujeres tuvieron una tasa de supervivencia del **74.20%** ($n = 233$ de 314), frente al **18.89%** ($n = 109$ de 577) en los hombres.
   - La prueba de independencia chi-cuadrado ($\chi^2 = 260.72$, $p = 1.20 \times 10^{-58}$, $gl = 1$) confirma que esta asociación es estadísticamente abrumadora y no atribuible al azar muestral. Refleja la aplicación estricta de la directiva de evacuación *"mujeres y niños primero"*.

2. **Estratificación socioeconómica (Clase del tiquete):**
   - **Primera Clase:** Supervivencia del **62.96%** ($n = 136$ de 216).
   - **Segunda Clase:** Supervivencia del **47.28%** ($n = 87$ de 184).
   - **Tercera Clase:** Supervivencia del **24.24%** ($n = 119$ de 491).
   - El acceso físico a las cubiertas superiores de botes salvavidas y la ubicación espacial de los camarotes favoreció notablemente a las clases altas.

3. **Interacción crítica: Género $\times$ Clase Social:**
   - Mujeres de 1ra y 2da clase alcanzaron tasas de supervivencia casi totales: **96.81%** ($n = 91$ de 94) en 1ra clase y **92.11%** ($n = 70$ de 76) en 2da clase.
   - En contraste, las mujeres de 3ra clase vieron reducida su probabilidad al **50.00%** ($n = 72$ de 144).
   - Para los hombres, viajar en 1ra clase elevó la supervivencia al **36.89%** ($n = 45$ de 122), comparado con solo **15.74%** ($n = 17$ de 108) en 2da clase y **13.54%** ($n = 47$ de 347) en 3ra clase.

4. **Protección a menores de edad:**
   - Los pasajeros clasificados como niños (`child`, $< 16$ años con edad conocida) registraron una supervivencia del **59.04%** ($n = 49$ de 83), significativamente superior a la de los varones adultos (**16.50%**, $n = 90$ de 545).

5. **Composición familiar:**
   - Los pasajeros que viajaban solos registraron una supervivencia del **30.35%** ($n = 163$ de 537).
   - Viajar en familias pequeñas (1 a 3 acompañantes) aumentó la supervivencia al rango del **55%–57%**. Sin embargo, en familias numerosas ($\ge 4$ acompañantes, $n = 60$), la tasa cayó drásticamente por debajo del **20%**, debido a la dificultad logística de reunir y evacuar grupos grandes en medio del pánico.

---

## 2. Decisiones de Limpieza y Cuantificación de Sesgos

Toda estrategia de preparación de datos introduce supuestos metodológicos que deben ser explicitados:

1. **Definición de `Person` y sesgo por edades faltantes (Reto 8.1):**
   - Existen **177 pasajeros** sin edad registrada ($19.87\%$ de la muestra etiquetada).
   - Bajo la regla determinista vectorizada, un pasajero sin edad no cumple `Age < 16` y se clasifica automáticamente por su sexo biológico (`male` o `female`).
   - Si asumimos que entre los faltantes existía la misma proporción de niños que entre las edades conocidas ($\approx 11.62\%$, 83/714), **aproximadamente 21 niños fueron subestimados** y clasificados erróneamente como adultos, introduciendo un sesgo conservador en la tasa de niños.

2. **Imputación de `Age` por media condicional de género (Reto 22.1):**
   - La desviación estándar de la edad se redujo de **14.53 años** a **13.01 años** tras imputar los 177 faltantes con las medias por sexo ($30.73$ hombres, $27.92$ mujeres).
   - Este método comprime artificialmente la varianza y genera dos picos densos irreales en el histograma. Por esta razón, la imputación global se limita al EDA; para el modelado predictivo se preserva `AgeOriginal` para imputar estrictamente dentro del pliegue de entrenamiento sin fuga de datos.

3. **Consolidación de Títulos (`Master` $\to$ `Mr`):**
   - El título `Master` correspondía históricamente a niños varones. Al fusionarlo con `Mr` en `TTitulo`, se mezcla una subpoblación de alta prioridad de rescate con adultos masculinos de menor prioridad. Esta pérdida de señal categórica es compensada en el análisis mediante la columna derivada `Person` cuando la edad está disponible.

4. **Tratamiento de `Cabin` (Categoría explícita `'X'`):**
   - El 77.10% ($n = 687$) de los pasajeros no reporta cabina. Se asignó la categoría explícita `'X'` e indicador `HasCabin`. Tener cabina registrada se asocia con un 66.67% de supervivencia frente al 29.99% en quienes no la tienen, pero esto responde en gran medida a un artefacto de registro: los sobrevivientes tuvieron mayor oportunidad de registrar su camarote tras el desastre.

---

## 3. Confusiones Detectadas y Desacoplamiento

El análisis univariado sugería correlaciones engañosas que fueron aisladas mediante estratificación controlada:

1. **Efecto aparente del puerto de embarque (`Embarked`):**
   - Los pasajeros embarcados en Cherbourg ($C$) mostraron una supervivencia aparente del **55.36%** ($n = 93$ de 168), muy por encima de Southampton ($S$, **33.70%**) y Queenstown ($Q$, **38.96%**).
   - *Control por estratificación:* Al cruzar puerto con clase, se evidencia que en Cherbourg el **50.60%** viajaba en 1ra clase, comparado con solo el **19.74%** en Southampton y el **2.60%** en Queenstown. Al comparar pasajeros de una misma clase (ej. 3ra clase: $C = 37.88\%$, $S = 18.98\%$, $Q = 37.50\%$), la aparente ventaja global de Cherbourg se diluye, confirmando que la clase del tiquete operaba como variable de confusión principal.

2. **Tarifa pagada vs. Clase del tiquete:**
   - La tarifa (`Fare`) correlaciona positivamente con la supervivencia ($r = +0.26$).
   - Sin embargo, al segmentar la tarifa en cuartiles *dentro de cada clase* (`FareQ`), el efecto intra-clase es marginal comparado con el salto discreto entre 3ra, 2da y 1ra clase. La tarifa por sí misma actúa primordialmente como un proxy de la clase y la ubicación en la nave.

---

## 4. Limitaciones y Advertencias Metodológicas

> [!WARNING]
> **Lo que NO podemos afirmar a partir de este estudio:**
> 1. **Asociación no implica Causalidad:** Una correlación estadística alta (ej. ser mujer o tener tiquete caro) describe lo observado bajo el protocolo específico del 15 de abril de 1912; no garantiza que en un evento contemporáneo ocurra el mismo resultado si los procedimientos de evacuación modernos son distintos.
> 2. **Evento histórico único:** El Titanic constituye una muestra observacional de un único siniestro nocturno con tiempo prolongado de hundimiento ($\approx 2.5$ horas) y escasez de botes. No es representativo de naufragios con colapso súbito o incendios a bordo.
> 3. **Riesgo ético de discriminación en modelos de seguro:** Modelar el riesgo actuarial o la prima de un seguro utilizando directamente `Sex` o `Pclass` incurre en sesgos discriminatorios y regulatorios inaceptables en pólizas actuales.

