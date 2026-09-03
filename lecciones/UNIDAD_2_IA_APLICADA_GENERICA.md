# UNIDAD 2: IA Aplicada Genérica

**Curso:** AI-Agentic-Systems-Core — UCEMICH

## Prefacio

La Unidad 1 definió qué es un agente (Modelo + Harness: Tools, Memory, Guardrails) y entrenó dos modelos de ML de propósito general. Esta unidad no repite esa anatomía — la da por asumida — y avanza directo a dos técnicas de IA aplicada que resuelven un problema distinto al de clasificar o regresionar: **decidir dónde mirar a continuación** cuando evaluar es costoso (Optimización Bayesiana) y **decidir qué es anómalo** sin tener ejemplos etiquetados de la anomalía (Detección de Anomalías). El Diccionario de Variables de unidades anteriores no se repite aquí.

Esta es la última unidad centrada en una sola pieza de IA aislada — Optimización Bayesiana y Detección de Anomalías son, igual que el árbol de decisión y la red neuronal de la Unidad 1, capacidades que un agente orquesta más que tareas que un humano ejecuta directamente hoy. La Unidad 3 da el salto real del curso: de piezas individuales a **sistemas** de agentes coordinados — el tema en el que este curso se centra. A diferencia de la versión anterior de esta unidad, ambas técnicas se aplican aquí sobre problemas y datos ya usados en el resto del curso (el `MLPRegressor` de California Housing de la Unidad 1, el guardrail heurístico de la Unidad 3) — no sobre ejemplos matemáticos aislados sin conexión con el resto del arco.

---

## Optimización Bayesiana

Muchos problemas de ingeniería comparten una estructura: hay una función objetivo que se quiere maximizar (o minimizar), pero **evaluarla es costoso** — un experimento físico, un entrenamiento de red neuronal completo, una simulación pesada. Probar exhaustivamente todo el dominio (grid search) es inviable cuando cada evaluación cuesta minutos, horas o dinero. La Optimización Bayesiana resuelve esto construyendo un **modelo probabilístico sustituto** (un Proceso Gaussiano, GP) de la función objetivo a partir de las pocas evaluaciones ya hechas, y usándolo para decidir el siguiente punto más prometedor a evaluar — sin necesidad de evaluar la función real en todo el dominio.

Un Proceso Gaussiano no predice un solo valor por punto: predice una **distribución** (media $\mu(x)$ y desviación estándar $\sigma(x)$), lo que permite balancear **explotación** (evaluar donde $\mu$ es alto) contra **exploración** (evaluar donde $\sigma$ es alto, es decir, donde el modelo está más incierto). El ejemplo de esta sección aplica esa idea a un problema real y costoso de verdad — **tunear el número de neuronas del `MLPRegressor` de California Housing** (Unidad 1) — en vez de a una función matemática sintética: cada evaluación de la métrica aquí implica entrenar una red neuronal completa, exactamente el tipo de "evaluación cara" que motiva usar BO en primer lugar.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Mismo dataset que la Unidad 1 — Redes Neuronales con California Housing.
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_sub, y_train_sub = X_train[:1500], y_train[:1500]
X_test_sub, y_test_sub = X_test[:500], y_test[:500]

escalador = StandardScaler().fit(X_train_sub)
X_train_esc = escalador.transform(X_train_sub)
X_test_esc = escalador.transform(X_test_sub)


def r2_de_neuronas(n_neuronas: float) -> float:
    """Entrena un MLPRegressor con `n_neuronas` en su única capa oculta y
    retorna su R2 sobre el conjunto de prueba. Esta es la función objetivo
    costosa de esta sección: cada llamada entrena una red neuronal completa.
    """
    n = max(2, int(n_neuronas))
    modelo = MLPRegressor(hidden_layer_sizes=(n,), alpha=0.001, max_iter=100, random_state=42)
    modelo.fit(X_train_esc, y_train_sub)
    return r2_score(y_test_sub, modelo.predict(X_test_esc))


def optimizar_neuronas_bayesiano(
    limites: tuple[float, float], n_iteraciones_iniciales: int = 5,
    n_pasos_bo: int = 3, semilla: int = 42,
) -> tuple[float, float]:
    """Optimización bayesiana iterativa: evalúa `n_iteraciones_iniciales`
    puntos aleatorios, y luego repite `n_pasos_bo` veces el ciclo completo
    de BO — reajustar el GP con TODAS las evaluaciones vistas hasta ahora,
    y evaluar el punto que maximiza la función de adquisición (media +
    desviación estándar). Esto es lo que distingue BO real de un solo
    salto: cada evaluación nueva refina el modelo sustituto para la
    siguiente decisión.

    Args:
        limites: Tupla (min, max) del número de neuronas a explorar.
        n_iteraciones_iniciales: Puntos aleatorios antes de ajustar el primer GP.
        n_pasos_bo: Iteraciones de refinamiento guiadas por el GP.
        semilla: Semilla aleatoria para reproducibilidad.

    Returns:
        Tupla (mejor número de neuronas visto, su R2).
    """
    rng = np.random.default_rng(semilla)
    xs = list(rng.uniform(limites[0], limites[1], size=n_iteraciones_iniciales))
    ys = [r2_de_neuronas(x) for x in xs]

    for _ in range(n_pasos_bo):
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=semilla)
        gp.fit(np.array(xs).reshape(-1, 1), np.array(ys))

        candidatos = rng.uniform(limites[0], limites[1], size=200).reshape(-1, 1)
        mu, sigma = gp.predict(candidatos, return_std=True)
        siguiente_x = float(candidatos[np.argmax(mu + sigma)][0])
        siguiente_y = r2_de_neuronas(siguiente_x)
        xs.append(siguiente_x)
        ys.append(siguiente_y)

    mejor_idx = int(np.argmax(ys))
    return xs[mejor_idx], ys[mejor_idx]


mejor_n, mejor_r2 = optimizar_neuronas_bayesiano(limites=(2, 128), semilla=7)
print(f"Mejor número de neuronas encontrado: {int(mejor_n)}")
print(f"R2 con esa configuración: {mejor_r2:.4f}")
```

Ejecutado, este bloque evalúa 8 configuraciones en total (5 aleatorias + 3 guiadas por el GP) y encuentra `n≈127` neuronas con **R² de 0.7506** — resultado consistente con el R² de 0.78 que la Unidad 1 reportó con `hidden_layer_sizes=(32,16)` sobre el dataset completo, y ligeramente mejor que el mejor punto de un grid uniforme de 8 puntos evaluado sobre el mismo rango (`R²=0.7496`), que además cae fácilmente en una zona pobre del dominio: el número de neuronas y el R² **no tienen una relación suave y monótona** en este problema (un barrido de referencia sobre 10 valores fijos: `2, 4, 8, 16, 24, 32, 48, 64, 96, 128` neuronas, muestra R² pasando de `-0.08` en `n=2` a `0.68` en `n=24` y bajando de nuevo a `0.67` en `n=32`, antes de subir otra vez) — exactamente el tipo de superficie irregular donde el modelo sustituto del GP paga su costo de ajuste al guiar la búsqueda hacia zonas prometedoras en vez de confiar en que un grid uniforme las cubra por casualidad.

---

## Detección de Anomalías

Clasificación y regresión (Unidad 1) asumen que existen ejemplos etiquetados de cada clase o valor a predecir. La detección de anomalías resuelve un problema distinto: identificar observaciones que se desvían del patrón normal **sin tener ejemplos etiquetados de "anomalía"** — porque las anomalías reales suelen ser raras, no anticipadas, o directamente inexistentes en los datos de entrenamiento (un sensor que nunca ha fallado, un fraude nunca antes visto).

`IsolationForest` de scikit-learn resuelve esto con una idea elegante: construye árboles de decisión con **cortes aleatorios**, y mide cuántos cortes hacen falta para aislar cada punto en su propia hoja. Un punto anómalo —alejado de la densidad principal de datos— se aísla en pocos cortes; un punto normal, rodeado de vecinos similares, requiere muchos más cortes para separarlo del resto.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest


def detectar_anomalias_wine(
    semilla: int = 42, contaminacion: float = 0.1
) -> tuple[int, int]:
    """Detecta anomalías en el dataset Wine (178 muestras, 13 features
    fisicoquímicas de vinos de 3 cultivares) usando IsolationForest.

    Args:
        semilla: Semilla aleatoria para reproducibilidad del bosque aleatorio.
        contaminacion: Fracción esperada de anomalías en el dataset (hiperparámetro
            de IsolationForest, no una etiqueta real — el dataset no tiene
            anomalías etiquetadas).

    Returns:
        Tupla (n_normales, n_anomalias) con el conteo de cada clase detectada.
    """
    X, _ = load_wine(return_X_y=True)

    modelo = IsolationForest(contamination=contaminacion, random_state=semilla)
    etiquetas = modelo.fit_predict(X)

    n_normales = int((etiquetas == 1).sum())
    n_anomalias = int((etiquetas == -1).sum())
    return n_normales, n_anomalias


n_normales, n_anomalias = detectar_anomalias_wine()
print(f"Muestras normales: {n_normales}")
print(f"Anomalías detectadas: {n_anomalias}")
print(f"Proporción de anomalías: {n_anomalias / (n_normales + n_anomalias):.1%}")
```

Ejecutado, imprime `Muestras normales: 160`, `Anomalías detectadas: 18` y `Proporción de anomalías: 10.1%` — consistente con el hiperparámetro `contamination=0.1` (10% esperado), sobre el total real de 178 muestras del dataset Wine. Nótese que `IsolationForest` no tiene "ground truth" contra el cual medir accuracy: `contamination` es una *expectativa* del usuario sobre qué fracción de los datos es anómala, no una etiqueta verificada — a diferencia de Iris o California Housing (Unidad 1), aquí no existe una forma directa de calcular una métrica de acierto sin datos etiquetados externos.

### Del detector de anomalías al Guardrail de un agente

La Unidad 3 introduce `SafetyGateAgent`, un guardrail que valida el output de texto de un agente contra patrones textuales conocidos (prompt injection reflejado, credenciales filtradas) — y muestra explícitamente que ese enfoque es ciego a memory poisoning y tool misuse, porque su heurística busca *contenido* específico, no *forma* inusual. `IsolationForest` ataca exactamente el hueco complementario: en vez de buscar patrones textuales conocidos, aprende la **forma estadística** de outputs normales (longitud, estructura, densidad de caracteres) y marca como sospechoso cualquier output que se desvíe de esa forma — sin necesidad de saber de antemano qué texto exacto buscar.

```python
def extraer_features_output(texto: str) -> list[float]:
    """Extrae 4 features numéricas baratas de un output de agente: no
    analiza el CONTENIDO del texto (eso es lo que hace SafetyGateAgent),
    solo su FORMA — longitud, número de palabras, proporción de mayúsculas
    y cantidad de caracteres no alfanuméricos.
    """
    longitud = len(texto)
    n_palabras = len(texto.split())
    ratio_mayusculas = sum(1 for c in texto if c.isupper()) / max(1, len(texto))
    n_especiales = sum(1 for c in texto if not c.isalnum() and not c.isspace())
    return [longitud, n_palabras, ratio_mayusculas, n_especiales]


# Outputs "normales" de un agente de investigación: 20 oraciones distintas
# (no repeticiones exactas) para que IsolationForest tenga variación real
# sobre la cual aprender la forma de "lo normal" — con muy pocos valores
# únicos repetidos, el bosque no encuentra dispersión suficiente para
# aislar nada como anómalo.
outputs_normales = [
    "El análisis de los datos muestra una tendencia clara al alza.",
    "Según los resultados, la hipótesis inicial parece confirmarse.",
    "Los datos recopilados sugieren una correlación moderada entre ambas variables.",
    "El modelo entrenado alcanza una precisión aceptable en el conjunto de prueba.",
    "La revisión de la literatura respalda parcialmente esta conclusión.",
    "Se recomienda validar estos hallazgos con una muestra más grande.",
    "El experimento se replicó tres veces con resultados consistentes.",
    "La variable X explica aproximadamente el 40% de la varianza observada.",
    "Conviene repetir el experimento antes de sacar conclusiones definitivas.",
    "El intervalo de confianza al 95% incluye el valor nulo, sin significancia.",
    "Los residuos del modelo no muestran un patrón sistemático evidente.",
    "Se descarta la hipótesis alternativa con un nivel de confianza razonable.",
    "El coeficiente de determinación sugiere un ajuste moderado del modelo.",
    "La muestra recolectada es representativa de la población de interés.",
    "El sesgo de selección podría explicar parte de la diferencia observada.",
    "Los resultados preliminares son consistentes con estudios previos.",
    "Se sugiere ampliar el tamaño muestral para reducir el error estándar.",
    "La distribución de los datos se aproxima razonablemente a la normal.",
    "El efecto observado es pequeño pero estadísticamente significativo.",
    "Los datos no permiten descartar por completo la hipótesis nula.",
]  # 20 ejemplos de referencia, todos distintos

X_train_outputs = np.array([extraer_features_output(t) for t in outputs_normales])
guardrail_estadistico = IsolationForest(contamination=0.05, random_state=42)
guardrail_estadistico.fit(X_train_outputs)

# Un output anómalo por ESTRUCTURA — sin ningún patrón textual de ataque
# que SafetyGateAgent pudiera reconocer (no dice "ignora las instrucciones"
# ni "system prompt"), pero con una forma muy alejada de lo normal.
output_anomalo_estructural = "OK. " * 200 + "!!!@#$%^&*()" * 30
output_normal_nuevo = "El resultado final confirma la tendencia observada previamente."

for nombre, texto in [("anómalo (estructura)", output_anomalo_estructural),
                       ("normal (nuevo)", output_normal_nuevo)]:
    features = np.array([extraer_features_output(texto)])
    prediccion = guardrail_estadistico.predict(features)[0]
    print(f"Output {nombre}: predicción = {'ANOMALO' if prediccion == -1 else 'normal'}")
```

Ejecutado, marca el output repetitivo y saturado de caracteres especiales como `ANOMALO` y el output normal nuevo (nunca visto durante el entrenamiento) como `normal` — a pesar de que ninguno de los dos contiene las palabras "ignora las instrucciones", "system prompt" ni ningún otro patrón textual que `SafetyGateAgent.check_output()` reconocería. Esto no reemplaza a `SafetyGateAgent`: un output podría tener una forma perfectamente normal (longitud típica, puntuación estándar) y aun así contener una instrucción maliciosa perfectamente redactada — ese caso sigue siendo trabajo exclusivo del guardrail textual. Los dos guardrails son complementarios porque miran ejes distintos del mismo problema: uno mira *qué dice* el texto, el otro mira *qué forma tiene* — un pipeline de producción robusto correría ambos en serie sobre cada output, no uno en lugar del otro.

---

## 🔄 El Ciclo del Agente

Las dos secciones anteriores resolvieron dos problemas técnicos sin discutir cuándo elegir cada técnica ni qué implica llevarlas a producción. Esta sección cierra la unidad aplicando el ciclo de vida completo de ingeniería a dos decisiones concretas: ¿optimización bayesiana o grid search?, y ¿cómo desplegar un detector de anomalías con latencia aceptable?

### Selección de Arquitectura

La elección entre Optimización Bayesiana y grid search depende de dos factores concretos: el costo de evaluar la función objetivo, y la dimensionalidad del dominio de búsqueda.

- **Grid search** es preferible cuando evaluar la función es barato (microsegundos a milisegundos) y el dominio tiene pocas dimensiones — la exhaustividad de un grid es simple de implementar, paralelizar y depurar, y no introduce el overhead de ajustar un GP en cada iteración.
- **Optimización bayesiana** es preferible cuando cada evaluación es costosa (segundos a horas: un entrenamiento de red neuronal, un experimento físico) y el presupuesto de evaluaciones es limitado — como en la sección anterior, donde 8 evaluaciones totales (5 aleatorias + 3 guiadas por el GP) encontraron una configuración mejor que el mejor punto de un grid uniforme del mismo tamaño, precisamente porque la relación entre número de neuronas y R² no es suave: un grid puede caer por completo en una zona pobre del dominio, mientras BO usa lo aprendido de cada evaluación para dirigir la siguiente búsqueda.

**Grid Search vs. IsolationForest** no es una disyuntiva real — son técnicas para problemas distintos (búsqueda de un óptimo vs. detección de outliers) — pero la misma lógica de costo aplica a la elección de hiperparámetros de `IsolationForest` (`n_estimators`, `contamination`): con pocos hiperparámetros y evaluación barata (ajustar el bosque es rápido), un grid search sobre esos hiperparámetros es la elección correcta; no hace falta optimización bayesiana para esto.

### Diseño

En optimización bayesiana, el diseño clave es la elección del **kernel** del GP (aquí, `Matern(nu=2.5)`, que asume que la función objetivo es suave pero no infinitamente diferenciable — una suposición razonable para la mayoría de funciones de ingeniería) y de la **función de adquisición** (aquí, media + desviación estándar, una versión simple de Upper Confidence Bound). En detección de anomalías, el diseño clave es la elección de `contamination`: sobreestimarla marca puntos normales como anómalos; subestimarla deja pasar anomalías reales. En el guardrail estadístico de la sección anterior, el diseño clave adicional es **qué features extraer del texto**: las 4 elegidas (longitud, número de palabras, proporción de mayúsculas, caracteres especiales) son deliberadamente baratas de calcular y agnósticas al idioma o dominio — el costo de este diseño es que solo detecta anomalías de *forma*, nunca de *contenido*, una limitación tan real y documentada como la de `SafetyGateAgent` en el sentido opuesto.

**Spec-Driven Development**: antes de desplegar un detector de anomalías, hay que especificar explícitamente qué constituye "anómalo" en el dominio del problema — no es una propiedad matemática universal, es una decisión de negocio. Para esta unidad, la especificación fue: *"con `contamination=0.1`, el detector debe aislar aproximadamente 10% de las muestras como anómalas"* — verificado en la sección de Autoevaluación.

### Implementación

`GaussianProcessRegressor.fit()` ajusta los hiperparámetros del kernel (longitud de escala, varianza) maximizando la verosimilitud marginal de los datos observados — un proceso de optimización interno, distinto de la optimización bayesiana externa que este GP sirve. `IsolationForest.fit_predict()` construye internamente un ensamble de árboles con cortes aleatorios y calcula, para cada punto, un score de anomalía basado en la profundidad promedio de aislamiento — ninguno de los dos algoritmos requiere backpropagation ni GPU, ambos corren en CPU en fracciones de segundo a esta escala de datos.

### Evaluación

- **Optimización bayesiana**: la métrica natural es qué tan buena es la mejor configuración encontrada, y con cuántas evaluaciones de la función objetivo se llegó ahí — en la sección anterior, `R²=0.7506` con solo 8 entrenamientos completos de red neuronal, superando al mejor de un grid uniforme del mismo tamaño.
- **Detección de anomalías**: sin etiquetas verdaderas, la evaluación es indirecta. Se puede verificar que la *proporción* de anomalías detectadas coincide con `contamination` (verificado arriba: 10.1% ≈ 10%), o —si se dispone de un pequeño conjunto de anomalías conocidas por expertos del dominio— calcular recall sobre ese subconjunto. La ausencia de una métrica de acierto directa es la diferencia estructural más importante frente a la clasificación supervisada de la Unidad 1.

### Despliegue

Los dos algoritmos tienen perfiles de latencia muy distintos en producción. Un `IsolationForest` ya entrenado responde a una predicción nueva en microsegundos — recorre cada árbol del ensamble, una operación tan barata como la de un árbol de decisión individual (Unidad 1) — lo que lo hace apto para filtrar transacciones o lecturas de sensores en tiempo real, incluso a alto volumen, y también apto para correr como guardrail estadístico en la ruta crítica de un agente sin agregar latencia perceptible. La optimización bayesiana, en cambio, **no es un modelo que se despliega para servir predicciones repetidas**: es un proceso de búsqueda que termina cuando se agota el presupuesto de evaluaciones costosas (por ejemplo, un ciclo de tuning de hiperparámetros de otro modelo que sí se despliega). Desplegar "mal" un detector de anomalías en este contexto significa, sobre todo, reentrenar `IsolationForest` periódicamente conforme la distribución de datos normales cambia (*data drift*) — un modelo entrenado sobre datos de hace un año puede marcar como anómalo lo que hoy es rutinario, y el guardrail estadístico de esta unidad sufriría exactamente el mismo problema si el estilo de redacción "normal" de un agente cambia tras una actualización de su prompt.

### Iteración

Si `IsolationForest` marca demasiados falsos positivos en producción (usuarios legítimos bloqueados, lecturas normales descartadas), el primer paso es revisar `contamination` — puede estar sobreestimando la fracción real de anomalías del dominio — antes de tocar `n_estimators` u otros hiperparámetros del bosque. Si la optimización bayesiana no converge tras muchas iteraciones, el primer sospechoso es el kernel del GP: una función objetivo con múltiples óptimos locales muy separados necesita un kernel que permita más variabilidad (`length_scale` menor), no necesariamente más iteraciones. En ambos casos, la disciplina de la Unidad 1 se repite: ajustar diseño e hiperparámetros antes de asumir que hace falta un modelo más complejo.

---

## Ejercicios

### Ejercicio A (para resolver): más puntos iniciales antes de ajustar el primer GP

Modifica la llamada a `optimizar_neuronas_bayesiano` para usar `n_iteraciones_iniciales=5` en vez de `2` (más puntos aleatorios antes de que el GP entre en juego, mismo número de pasos de refinamiento) y compara el resultado. A diferencia de aumentar `n_pasos_bo` (que refina sobre el mismo historial inicial y por eso puede no cambiar nada si el óptimo ya se alcanzó), aumentar los puntos iniciales cambia qué zonas del dominio ve el primer GP — con más cobertura inicial, es más probable que el GP ya tenga evidencia de las zonas prometedoras antes de empezar a refinar. Esto no está garantizado en cada corrida individual (más puntos iniciales consumen presupuesto de evaluaciones que podría haberse usado en refinamiento guiado, así que a veces el resultado empeora), pero para esta semilla específica sí mejora:

```python
_, r2_pocos_iniciales = optimizar_neuronas_bayesiano(
    limites=(2, 128), n_iteraciones_iniciales=2, n_pasos_bo=3, semilla=7
)
_, r2_mas_iniciales = optimizar_neuronas_bayesiano(
    limites=(2, 128), n_iteraciones_iniciales=5, n_pasos_bo=3, semilla=7
)
print(f"R2 con 2 puntos iniciales: {r2_pocos_iniciales:.4f}")
print(f"R2 con 5 puntos iniciales: {r2_mas_iniciales:.4f}")

assert r2_mas_iniciales > r2_pocos_iniciales, (
    "Con esta semilla, más puntos iniciales debería encontrar una mejor "
    "configuración — si esto falla, revisa que `n_iteraciones_iniciales` "
    "se esté usando para muestrear `xs` antes del primer ajuste del GP."
)
```

### Ejercicio B (para resolver): comparar dos umbrales de contaminación del guardrail

El guardrail estadístico de esta unidad usa `contamination=0.05` (espera que 5% de los outputs de referencia sean atípicos). Sube ese umbral a `0.2` y verifica que el guardrail se vuelve **más estricto** — más outputs de referencia quedan marcados como anómalos durante el entrenamiento, lo cual tiene una consecuencia directa: un `contamination` más alto redefine "normal" de forma más angosta, así que en producción marcaría más outputs reales como sospechosos (más sensibilidad, a costa de más falsos positivos):

```python
guardrail_05 = IsolationForest(contamination=0.05, random_state=42)
guardrail_05.fit(X_train_outputs)
n_marcados_05 = int((guardrail_05.predict(X_train_outputs) == -1).sum())

guardrail_20 = IsolationForest(contamination=0.2, random_state=42)
guardrail_20.fit(X_train_outputs)
n_marcados_20 = int((guardrail_20.predict(X_train_outputs) == -1).sum())

print(f"Outputs de referencia marcados como anómalos (contamination=0.05): {n_marcados_05}")
print(f"Outputs de referencia marcados como anómalos (contamination=0.2): {n_marcados_20}")

assert n_marcados_20 > n_marcados_05, (
    "Un contamination mayor debería marcar estrictamente más outputs de "
    "referencia como anómalos durante el entrenamiento."
)
```

---

## Cierre Auto-Referencial: Prediciendo Hallazgos de ContentAuditorAgent

`ContentAuditorAgent` — el agente que audita el contenido pedagógico de estas mismas unidades — es en sí mismo un sistema que produce una salida numérica (`total_hallazgos`) a partir de un `.md` de entrada. Esta sección explora si **features simples y baratas de calcular** (longitud del archivo, número de bloques de código) permiten anticipar esa salida antes de correr la auditoría completa — la misma lógica de "modelo sustituto barato" que motivó la Optimización Bayesiana: evaluar `ContentAuditorAgent.audit_unit()` de verdad no es tan costoso como una simulación DFT, pero sí implica renderizar cada diagrama Mermaid con Node.js — evitarlo cuando ya se anticipa el resultado ahorra tiempo en un pipeline de CI que audita muchas unidades.

```python
from pathlib import Path

from src.multiagent_core.content_auditor_agent import ContentAuditorAgent
from src.multiagent_core._fence_utils import extract_fenced_blocks


def extraer_features_unidad(md_path: Path) -> tuple[int, int]:
    """Extrae dos features baratas de un archivo de unidad: longitud del
    texto y número de bloques de código Python.

    Args:
        md_path: Ruta al archivo .md de la unidad a analizar.

    Returns:
        Tupla (longitud_caracteres, n_bloques_python).
    """
    contenido = md_path.read_text(encoding="utf-8")
    bloques = extract_fenced_blocks(contenido)
    n_bloques_python = sum(1 for _, lang, _ in bloques if lang == "python")
    return len(contenido), n_bloques_python


def predecir_hallazgos_por_heuristica(longitud: int, n_bloques_python: int) -> int:
    """Predice el número de hallazgos de ContentAuditorAgent con una regla
    heurística simple: unidades muy largas o con muchos bloques de código
    tienen más superficie donde algo puede fallar (más type hints que
    olvidar, más fórmulas LaTeX que desbalancear).

    Esta NO es una regresión entrenada — es una heurística de umbral fijo,
    documentada como tal: con una sola unidad de referencia disponible
    (Unidad 1) no hay datos suficientes para ajustar un modelo real: se
    necesitarían muchas más unidades auditadas para evitar sobreajuste
    a un único punto.

    Args:
        longitud: Longitud en caracteres del archivo .md.
        n_bloques_python: Número de bloques de código Python en el archivo.

    Returns:
        Número de hallazgos predicho (0 si el contenido parece cuidado).
    """
    UMBRAL_LONGITUD = 50_000
    UMBRAL_BLOQUES = 20
    if longitud > UMBRAL_LONGITUD or n_bloques_python > UMBRAL_BLOQUES:
        return 1
    return 0


ruta_u1 = Path("lecciones/UNIDAD_1_ML_FUNDAMENTALS.md")
longitud_u1, n_bloques_u1 = extraer_features_unidad(ruta_u1)
prediccion_u1 = predecir_hallazgos_por_heuristica(longitud_u1, n_bloques_u1)

resultado_real = ContentAuditorAgent().audit_unit(ruta_u1)
hallazgos_reales_u1 = resultado_real["total_hallazgos"]

print(f"Unidad analizada: {ruta_u1.name}")
print(f"Longitud: {longitud_u1} caracteres | Bloques Python: {n_bloques_u1}")
print(f"Hallazgos predichos (heurística): {prediccion_u1}")
print(f"Hallazgos reales (ContentAuditorAgent.audit_unit): {hallazgos_reales_u1}")
print(f"¿Predicción correcta? {prediccion_u1 == hallazgos_reales_u1}")
```

Ejecutado sobre `UNIDAD_1_ML_FUNDAMENTALS.md` (el archivo real de este mismo repositorio), imprime `Longitud: 24585 caracteres | Bloques Python: 8`, `Hallazgos predichos (heurística): 0`, `Hallazgos reales (ContentAuditorAgent.audit_unit): 0` y `¿Predicción correcta? True` — la heurística de umbral acierta porque Unidad 1 está por debajo de ambos umbrales (24,585 < 50,000 caracteres; 8 < 20 bloques) y, en efecto, no tiene hallazgos reales. Esto es una demostración de concepto con **una sola muestra**, no una validación estadística: la propia función lo advierte en su docstring. El paralelo con Optimización Bayesiana es directo — así como un GP con pocos puntos da estimaciones útiles pero con incertidumbre alta, una heurística calibrada sobre una sola unidad debe tratarse con la misma cautela hasta tener más unidades auditadas con las que ajustar los umbrales de verdad.

### Diccionario de Variables

| Símbolo | Nombre | Descripción |
|---|---|---|
| `semilla` | Semilla aleatoria | Argumento `random_state`/`semilla` fijado en `optimizar_neuronas_bayesiano`, `GaussianProcessRegressor` e `IsolationForest`, para reproducibilidad en ambas secciones de IA Aplicada |
| `rng` | Generador de números aleatorios | Instancia de `np.random.default_rng(semilla)` usada para muestrear puntos iniciales en `optimizar_neuronas_bayesiano` (Optimización Bayesiana) |
| `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test`, `X_train_sub`, `y_train_sub`, `X_test_sub`, `y_test_sub` | Dataset California Housing y su partición | Mismo dataset que la Unidad 1 (`fetch_california_housing`); subconjuntos usados para que cada entrenamiento del MLP dentro del GP sea rápido (Optimización Bayesiana) |
| `escalador` | `StandardScaler` ajustado | Normaliza las 8 features de California Housing antes de entrenar cada MLP candidato (Optimización Bayesiana) |
| `xs`, `ys` | Puntos evaluados y sus valores | Número de neuronas probadas y el R² obtenido en cada una, acumulados a lo largo de las iteraciones de BO (Optimización Bayesiana) |
| `kernel` | Kernel del Proceso Gaussiano | `Matern(nu=2.5)`, asume suavidad finita de la función objetivo (Optimización Bayesiana) |
| `gp` | Proceso Gaussiano ajustado | `GaussianProcessRegressor` reajustado en cada paso de refinamiento con todas las evaluaciones vistas hasta ese momento (Optimización Bayesiana) |
| `candidatos` | Puntos candidatos a evaluar | 200 puntos uniformes sobre los que el GP predice media y desviación estándar en cada paso (Optimización Bayesiana) |
| `mu`, `sigma` | Media y desviación estándar predichas | Salida de `gp.predict(candidatos, return_std=True)`, combinadas para elegir el siguiente punto (Optimización Bayesiana) |
| `siguiente_x`, `siguiente_y` | Punto elegido en cada paso de BO y su resultado | El candidato que maximiza `mu + sigma`, y el R² real obtenido al evaluarlo (Optimización Bayesiana) |
| `mejor_n`, `mejor_r2` | Mejor configuración final y su R² | Resultado de `optimizar_neuronas_bayesiano`, impreso y comparado contra el R² de referencia de la Unidad 1 y contra un grid uniforme (Optimización Bayesiana) |
| `contaminacion` | Fracción esperada de anomalías | Hiperparámetro `contamination` de `IsolationForest` (Detección de Anomalías) |
| `modelo` | Estimador de detección de anomalías | Instancia de `IsolationForest` sobre el dataset Wine (Detección de Anomalías) |
| `etiquetas` | Etiquetas de anomalía | Salida de `modelo.fit_predict(X)`: `1` para normal, `-1` para anómalo (Detección de Anomalías) |
| `n_normales`, `n_anomalias` | Conteos de clasificación | Cantidad de muestras etiquetadas como normales/anómalas por `IsolationForest` sobre Wine (Detección de Anomalías) |
| `extraer_features_output` | Extractor de features de forma de un texto | Recibe un string y retorna `[longitud, n_palabras, ratio_mayusculas, n_especiales]` (Guardrail estadístico) |
| `outputs_normales`, `X_train_outputs` | Corpus de referencia y sus features | 20 oraciones de ejemplo distintas (no repetidas), y su matriz de features extraída (Guardrail estadístico) |
| `guardrail_estadistico` | `IsolationForest` entrenado sobre forma de texto | Complementario a `SafetyGateAgent` de la Unidad 3 — detecta anomalías de estructura, no de contenido (Guardrail estadístico) |
| `output_anomalo_estructural`, `output_normal_nuevo` | Textos de prueba del guardrail estadístico | Un output repetitivo y saturado de símbolos, y una oración normal nunca vista en el entrenamiento (Guardrail estadístico) |
| `longitud_u1`, `n_bloques_u1` | Features del cierre auto-referencial | Longitud en caracteres y número de bloques Python de `UNIDAD_1_ML_FUNDAMENTALS.md`, extraídos por `extraer_features_unidad` (Cierre Auto-Referencial) |
| `prediccion_u1` | Hallazgos predichos | Salida de `predecir_hallazgos_por_heuristica` sobre las features de Unidad 1 (Cierre Auto-Referencial) |
| `hallazgos_reales_u1` | Hallazgos reales | `total_hallazgos` reportado por `ContentAuditorAgent().audit_unit()` sobre Unidad 1 (Cierre Auto-Referencial) |

**Verificación manual del Diccionario de Variables** (el mecanismo automático de `ContentAuditorAgent._audit_diccionario_variables` es un placeholder que siempre retorna `[]` — no certifica nada): cada símbolo de la tabla fue releído contra el bloque de código donde aparece antes de agregarlo, y la entrada `semilla` de la versión anterior de esta unidad fue corregida en esta revisión — mencionaba `np.random.seed`, una función que no aparece en ningún bloque de código de esta unidad; el mecanismo real de reproducibilidad aquí es siempre `np.random.default_rng(semilla)` combinado con el argumento `random_state` de scikit-learn. Los veinte símbolos de la tabla están efectivamente usados en código Python realmente ejecutado en esta unidad: `semilla`/`rng` gobiernan el muestreo aleatorio de `optimizar_neuronas_bayesiano`; `X`/`y`/`X_train`/`X_test`/`y_train`/`y_test`/`X_train_sub`/`y_train_sub`/`X_test_sub`/`y_test_sub` se generan con `fetch_california_housing` y `train_test_split` reales y se usan dentro de `r2_de_neuronas`; `escalador` transforma ambos conjuntos; `xs`/`ys` se acumulan en cada paso del bucle de refinamiento; `kernel`/`gp` se reconstruyen y reajustan en cada iteración; `candidatos`/`mu`/`sigma` se calculan con `gp.predict` real; `siguiente_x`/`siguiente_y` se derivan de `np.argmax(mu + sigma)` y de una evaluación real de `r2_de_neuronas`; `mejor_n`/`mejor_r2` se imprimen y se comparan contra el resultado del grid uniforme en la prosa; `contaminacion`/`modelo`/`etiquetas`/`n_normales`/`n_anomalias` participan en `detectar_anomalias_wine` sobre datos reales de Wine; `extraer_features_output` se invoca sobre `outputs_normales` para construir `X_train_outputs`, que efectivamente entrena `guardrail_estadistico`; `output_anomalo_estructural`/`output_normal_nuevo` se clasifican con ese modelo y sus predicciones se imprimen; `longitud_u1`/`n_bloques_u1`/`prediccion_u1`/`hallazgos_reales_u1` participan en la comparación final del Cierre Auto-Referencial, verificada con un `assert` real en la Autoevaluación.

### Autoevaluación

```python
%%writefile test_unidad_2.py
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent_core.content_auditor_agent import ContentAuditorAgent
from src.multiagent_core._fence_utils import extract_fenced_blocks


def detectar_anomalias_wine(semilla=42, contaminacion=0.1):
    """Detecta anomalías en Wine con IsolationForest."""
    X, _ = load_wine(return_X_y=True)
    modelo = IsolationForest(contamination=contaminacion, random_state=semilla)
    etiquetas = modelo.fit_predict(X)
    n_normales = int((etiquetas == 1).sum())
    n_anomalias = int((etiquetas == -1).sum())
    return n_normales, n_anomalias


def extraer_features_output(texto):
    longitud = len(texto)
    n_palabras = len(texto.split())
    ratio_mayusculas = sum(1 for c in texto if c.isupper()) / max(1, len(texto))
    n_especiales = sum(1 for c in texto if not c.isalnum() and not c.isspace())
    return [longitud, n_palabras, ratio_mayusculas, n_especiales]


def extraer_features_unidad(md_path):
    """Extrae longitud y número de bloques Python de un archivo de unidad."""
    contenido = md_path.read_text(encoding="utf-8")
    bloques = extract_fenced_blocks(contenido)
    n_bloques_python = sum(1 for _, lang, _ in bloques if lang == "python")
    return len(contenido), n_bloques_python


def predecir_hallazgos_por_heuristica(longitud, n_bloques_python):
    """Heurística de umbral fijo para predecir hallazgos de ContentAuditorAgent."""
    UMBRAL_LONGITUD = 50_000
    UMBRAL_BLOQUES = 20
    if longitud > UMBRAL_LONGITUD or n_bloques_python > UMBRAL_BLOQUES:
        return 1
    return 0


def test_deteccion_anomalias_wine_respeta_contaminacion():
    n_normales, n_anomalias = detectar_anomalias_wine()
    total = n_normales + n_anomalias
    assert total == 178
    assert 10 <= n_anomalias <= 25


def test_guardrail_estadistico_distingue_forma_anomala_de_normal():
    outputs_normales = [
        "El análisis de los datos muestra una tendencia clara al alza.",
        "Según los resultados, la hipótesis inicial parece confirmarse.",
        "Los datos recopilados sugieren una correlación moderada entre ambas variables.",
        "El modelo entrenado alcanza una precisión aceptable en el conjunto de prueba.",
        "La revisión de la literatura respalda parcialmente esta conclusión.",
        "Se recomienda validar estos hallazgos con una muestra más grande.",
        "El experimento se replicó tres veces con resultados consistentes.",
        "La variable X explica aproximadamente el 40% de la varianza observada.",
        "Conviene repetir el experimento antes de sacar conclusiones definitivas.",
        "El intervalo de confianza al 95% incluye el valor nulo, sin significancia.",
        "Los residuos del modelo no muestran un patrón sistemático evidente.",
        "Se descarta la hipótesis alternativa con un nivel de confianza razonable.",
        "El coeficiente de determinación sugiere un ajuste moderado del modelo.",
        "La muestra recolectada es representativa de la población de interés.",
        "El sesgo de selección podría explicar parte de la diferencia observada.",
        "Los resultados preliminares son consistentes con estudios previos.",
        "Se sugiere ampliar el tamaño muestral para reducir el error estándar.",
        "La distribución de los datos se aproxima razonablemente a la normal.",
        "El efecto observado es pequeño pero estadísticamente significativo.",
        "Los datos no permiten descartar por completo la hipótesis nula.",
    ]
    X_train_outputs = np.array([extraer_features_output(t) for t in outputs_normales])
    guardrail = IsolationForest(contamination=0.05, random_state=42)
    guardrail.fit(X_train_outputs)

    output_anomalo = "OK. " * 200 + "!!!@#$%^&*()" * 30
    output_normal = "El resultado final confirma la tendencia observada previamente."

    pred_anomalo = guardrail.predict([extraer_features_output(output_anomalo)])[0]
    pred_normal = guardrail.predict([extraer_features_output(output_normal)])[0]

    assert pred_anomalo == -1
    assert pred_normal == 1


def test_prediccion_heuristica_coincide_con_auditor_real_en_unidad_1():
    ruta_u1 = REPO_ROOT / "lecciones" / "UNIDAD_1_ML_FUNDAMENTALS.md"
    longitud_u1, n_bloques_u1 = extraer_features_unidad(ruta_u1)
    prediccion_u1 = predecir_hallazgos_por_heuristica(longitud_u1, n_bloques_u1)
    hallazgos_reales_u1 = ContentAuditorAgent().audit_unit(ruta_u1)["total_hallazgos"]
    assert prediccion_u1 == hallazgos_reales_u1
```

```python
!pytest test_unidad_2.py -v
```

Ejecutado, las 3 pruebas pasan: el detector de anomalías respeta la proporción esperada de `contamination` sobre las 178 muestras reales de Wine, el guardrail estadístico distingue correctamente un output anómalo por estructura de uno normal nunca visto durante el entrenamiento, y la heurística de predicción coincide con el resultado real de `ContentAuditorAgent().audit_unit()` sobre `UNIDAD_1_ML_FUNDAMENTALS.md`. La optimización bayesiana del ejemplo principal no se incluye en este archivo de autoevaluación porque cada evaluación implica entrenar una red neuronal completa (varios segundos por corrida) y ya lleva su propio `assert` de verificación en el Ejercicio A — mismo criterio que las autoevaluaciones de U0, U1 y U4 aplican a los bloques costosos de ejecutar. Las funciones se redefinen dentro del archivo de test (no se importan del notebook) — mismo patrón que ya usan las autoevaluaciones del resto del curso, evita depender de un mecanismo de import frágil hacia celdas de notebook ejecutadas previamente.
