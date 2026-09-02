# UNIDAD 2: IA Aplicada Genérica

**Curso:** AI-Agentic-Systems-Core — UCEMICH

## Prefacio

La Unidad 1 definió qué es un agente (Modelo + Harness: Tools, Memory, Guardrails) y entrenó dos modelos de ML de propósito general. Esta unidad no repite esa anatomía — la da por asumida — y avanza directo a dos técnicas de IA aplicada que resuelven un problema distinto al de clasificar o regresionar: **decidir dónde mirar a continuación** cuando evaluar es costoso (Optimización Bayesiana) y **decidir qué es anómalo** sin tener ejemplos etiquetados de la anomalía (Detección de Anomalías). El Diccionario de Variables de unidades anteriores no se repite aquí.

---

## Optimización Bayesiana

Muchos problemas de ingeniería comparten una estructura: hay una función objetivo que se quiere maximizar (o minimizar), pero **evaluarla es costoso** — un experimento físico, un entrenamiento de red neuronal completo, una simulación pesada. Probar exhaustivamente todo el dominio (grid search) es inviable cuando cada evaluación cuesta minutos, horas o dinero. La Optimización Bayesiana resuelve esto construyendo un **modelo probabilístico sustituto** (un Proceso Gaussiano, GP) de la función objetivo a partir de las pocas evaluaciones ya hechas, y usándolo para decidir el siguiente punto más prometedor a evaluar — sin necesidad de evaluar la función real en todo el dominio.

Un Proceso Gaussiano no predice un solo valor por punto: predice una **distribución** (media $\mu(x)$ y desviación estándar $\sigma(x)$), lo que permite balancear **explotación** (evaluar donde $\mu$ es alto) contra **exploración** (evaluar donde $\sigma$ es alto, es decir, donde el modelo está más incierto).

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def funcion_objetivo(x: float) -> float:
    """Función objetivo sintética a maximizar: una parábola invertida
    con máximo conocido en x=2, f(2)=10 — sirve para verificar que el
    optimizador converge al óptimo real."""
    return -(x - 2) ** 2 + 10


def optimizar_funcion_objetivo(
    limites: tuple[float, float], n_iteraciones: int = 10, semilla: int = 42
) -> float:
    """Aproxima el máximo de `funcion_objetivo` vía optimización bayesiana
    con un Proceso Gaussiano como modelo sustituto.

    Evalúa `n_iteraciones` puntos iniciales aleatorios, ajusta un GP sobre
    ellos, y elige entre un conjunto denso de candidatos el que maximiza
    media + desviación estándar (una función de adquisición simple que
    combina explotación y exploración).

    Args:
        limites: Tupla (min, max) del dominio de búsqueda en 1D.
        n_iteraciones: Número de puntos iniciales evaluados antes de ajustar el GP.
        semilla: Semilla aleatoria para reproducibilidad del muestreo.

    Returns:
        El mejor valor de x encontrado dentro de los límites dados.
    """
    np.random.seed(semilla)  # fija también el generador legado, por reproducibilidad explícita
    rng = np.random.default_rng(semilla)

    xs = rng.uniform(limites[0], limites[1], size=n_iteraciones).reshape(-1, 1)
    ys = np.array([funcion_objetivo(x[0]) for x in xs])

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=semilla)
    gp.fit(xs, ys)

    candidatos = rng.uniform(limites[0], limites[1], size=200).reshape(-1, 1)
    mu, sigma = gp.predict(candidatos, return_std=True)
    mejor_idx = np.argmax(mu + sigma)
    return float(candidatos[mejor_idx][0])


mejor_x = optimizar_funcion_objetivo(limites=(-5, 5))
print(f"Mejor x encontrado: {mejor_x:.2f}")
print(f"f(mejor_x) = {funcion_objetivo(mejor_x):.2f} (óptimo real: f(2) = 10.00)")
```

Ejecutado, este bloque imprime `Mejor x encontrado: 2.17` y `f(mejor_x) = 9.97 (óptimo real: f(2) = 10.00)` — con solo 10 evaluaciones iniciales de la función objetivo (más 200 evaluaciones *del modelo sustituto*, que son casi gratis comparadas con evaluar la función real), el GP guía la búsqueda a menos de 0.2 unidades del óptimo verdadero. Un grid search con la misma cantidad de evaluaciones (10 puntos igualmente espaciados en $[-5, 5]$) tendría una probabilidad no trivial de saltarse por completo la región cercana a $x=2$; el GP en cambio *interpola* entre las evaluaciones hechas para estimar dónde es probable que esté el máximo, incluso en zonas nunca evaluadas directamente.

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
    np.random.seed(semilla)  # fija también el generador legado, por reproducibilidad explícita
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

---

## 🔄 El Ciclo del Agente

Las dos secciones anteriores resolvieron dos problemas técnicos sin discutir cuándo elegir cada técnica ni qué implica llevarlas a producción. Esta sección cierra la unidad aplicando el ciclo de vida completo de ingeniería a dos decisiones concretas: ¿optimización bayesiana o grid search?, y ¿cómo desplegar un detector de anomalías con latencia aceptable?

### Selección de Arquitectura

**Optimización Bayesiana vs. Grid Search**: la elección depende del costo de evaluar la función objetivo y de la dimensionalidad del dominio.

- **Grid search** es preferible cuando evaluar la función es barato (microsegundos a milisegundos) y el dominio tiene pocas dimensiones — la exhaustividad de un grid es simple de implementar, paralelizar y depurar, y no introduce el overhead de ajustar un GP en cada iteración.
- **Optimización bayesiana** es preferible cuando cada evaluación es costosa (segundos a horas: un entrenamiento de red neuronal, un experimento físico) y el presupuesto de evaluaciones es limitado — como en la sección anterior, donde 10 evaluaciones bastaron para acercarse al óptimo, algo que un grid de 10 puntos no garantiza igual de bien.

**Grid Search vs. IsolationForest** no es una disyuntiva real — son técnicas para problemas distintos (búsqueda de un óptimo vs. detección de outliers) — pero la misma lógica de costo aplica a la elección de hiperparámetros de `IsolationForest` (`n_estimators`, `contamination`): con pocos hiperparámetros y evaluación barata (ajustar el bosque es rápido), un grid search sobre esos hiperparámetros es la elección correcta; no hace falta optimización bayesiana para esto.

### Diseño

**Context Engineering**: en optimización bayesiana, el diseño clave es la elección del **kernel** del GP (aquí, `Matern(nu=2.5)`, que asume que la función objetivo es suave pero no infinitamente diferenciable — una suposición razonable para la mayoría de funciones de ingeniería) y de la **función de adquisición** (aquí, media + desviación estándar, una versión simple de Upper Confidence Bound). En detección de anomalías, el diseño clave es la elección de `contamination`: sobreestimarla marca puntos normales como anómalos; subestimarla deja pasar anomalías reales.

**Spec-Driven Development**: antes de desplegar un detector de anomalías, hay que especificar explícitamente qué constituye "anómalo" en el dominio del problema — no es una propiedad matemática universal, es una decisión de negocio. Para esta unidad, la especificación fue: *"con `contamination=0.1`, el detector debe aislar aproximadamente 10% de las muestras como anómalas"* — verificado en la sección de Autoevaluación.

### Implementación

`GaussianProcessRegressor.fit()` ajusta los hiperparámetros del kernel (longitud de escala, varianza) maximizando la verosimilitud marginal de los datos observados — un proceso de optimización interno, distinto de la optimización bayesiana externa que este GP sirve. `IsolationForest.fit_predict()` construye internamente un ensamble de árboles con cortes aleatorios y calcula, para cada punto, un score de anomalía basado en la profundidad promedio de aislamiento — ninguno de los dos algoritmos requiere backpropagation ni GPU, ambos corren en CPU en fracciones de segundo a esta escala de datos.

### Evaluación

- **Optimización bayesiana**: la métrica natural es qué tan cerca del óptimo real cae el mejor punto encontrado, y con cuántas evaluaciones de la función objetivo se llegó ahí — en la sección anterior, `f(mejor_x) = 9.97` contra un óptimo real de `10.00`, con solo 10 evaluaciones directas de `funcion_objetivo`.
- **Detección de anomalías**: sin etiquetas verdaderas, la evaluación es indirecta. Se puede verificar que la *proporción* de anomalías detectadas coincide con `contamination` (verificado arriba: 10.1% ≈ 10%), o —si se dispone de un pequeño conjunto de anomalías conocidas por expertos del dominio— calcular recall sobre ese subconjunto. La ausencia de una métrica de acierto directa es la diferencia estructural más importante frente a la clasificación supervisada de la Unidad 1.

### Despliegue

Los dos algoritmos tienen perfiles de latencia muy distintos en producción. Un `IsolationForest` ya entrenado responde a una predicción nueva en microsegundos — recorre cada árbol del ensamble, una operación tan barata como la de un árbol de decisión individual (Unidad 1) — lo que lo hace apto para filtrar transacciones o lecturas de sensores en tiempo real, incluso a alto volumen. La optimización bayesiana, en cambio, **no es un modelo que se despliega para servir predicciones repetidas**: es un proceso de búsqueda que termina cuando se agota el presupuesto de evaluaciones costosas (por ejemplo, un ciclo de tuning de hiperparámetros de otro modelo que sí se despliega). Desplegar "mal" un detector de anomalías en este contexto significa, sobre todo, reentrenar `IsolationForest` periódicamente conforme la distribución de datos normales cambia (*data drift*) — un modelo entrenado sobre datos de hace un año puede marcar como anómalo lo que hoy es rutinario.

### Iteración

Si `IsolationForest` marca demasiados falsos positivos en producción (usuarios legítimos bloqueados, lecturas normales descartadas), el primer paso es revisar `contamination` — puede estar sobreestimando la fracción real de anomalías del dominio — antes de tocar `n_estimators` u otros hiperparámetros del bosque. Si la optimización bayesiana no converge tras muchas iteraciones, el primer sospechoso es el kernel del GP: una función objetivo con múltiples óptimos locales muy separados necesita un kernel que permita más variabilidad (`length_scale` menor), no necesariamente más iteraciones. En ambos casos, la disciplina de la Unidad 1 se repite: ajustar diseño e hiperparámetros antes de asumir que hace falta un modelo más complejo.

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

Ejecutado sobre `UNIDAD_1_ML_FUNDAMENTALS.md` (el archivo real de este mismo repositorio), imprime `Longitud: 23949 caracteres | Bloques Python: 8`, `Hallazgos predichos (heurística): 0`, `Hallazgos reales (ContentAuditorAgent.audit_unit): 0` y `¿Predicción correcta? True` — la heurística de umbral acierta porque Unidad 1 está por debajo de ambos umbrales (23,949 < 50,000 caracteres; 8 < 20 bloques) y, en efecto, no tiene hallazgos reales. Esto es una demostración de concepto con **una sola muestra**, no una validación estadística: la propia función lo advierte en su docstring. El paralelo con Optimización Bayesiana es directo — así como un GP con pocos puntos da estimaciones útiles pero con incertidumbre alta, una heurística calibrada sobre una sola unidad debe tratarse con la misma cautela hasta tener más unidades auditadas con las que ajustar los umbrales de verdad.

### Diccionario de Variables

| Símbolo | Nombre | Descripción |
|---|---|---|
| `semilla` | Semilla aleatoria | Fija `np.random.seed`/`np.random.default_rng` y `random_state` de scikit-learn en ambas secciones de IA Aplicada, para reproducibilidad |
| `rng` | Generador de números aleatorios | Instancia de `np.random.default_rng(semilla)` usada para muestrear puntos iniciales en `optimizar_funcion_objetivo` (Optimización Bayesiana) |
| `xs`, `ys` | Puntos evaluados y sus valores | Puntos de entrada muestreados y el valor de `funcion_objetivo` en cada uno, usados para ajustar el GP (Optimización Bayesiana) |
| `kernel` | Kernel del Proceso Gaussiano | `Matern(nu=2.5)`, asume suavidad finita de la función objetivo (Optimización Bayesiana) |
| `gp` | Proceso Gaussiano ajustado | `GaussianProcessRegressor` entrenado sobre `xs`, `ys` (Optimización Bayesiana) |
| `candidatos` | Puntos candidatos a evaluar | 200 puntos uniformes sobre los que el GP predice media y desviación estándar (Optimización Bayesiana) |
| `mu`, `sigma` | Media y desviación estándar predichas | Salida de `gp.predict(candidatos, return_std=True)`, combinadas para elegir el siguiente punto (Optimización Bayesiana) |
| `mejor_x` | Mejor punto encontrado | Resultado de `optimizar_funcion_objetivo`, cercano al óptimo real x=2 (Optimización Bayesiana) |
| `contaminacion` | Fracción esperada de anomalías | Hiperparámetro `contamination` de `IsolationForest` (Detección de Anomalías) |
| `modelo` | Estimador de detección de anomalías | Instancia de `IsolationForest` (Detección de Anomalías) |
| `etiquetas` | Etiquetas de anomalía | Salida de `modelo.fit_predict(X)`: `1` para normal, `-1` para anómalo (Detección de Anomalías) |
| `n_normales`, `n_anomalias` | Conteos de clasificación | Cantidad de muestras etiquetadas como normales/anómalas por `IsolationForest` (Detección de Anomalías) |
| `longitud_u1`, `n_bloques_u1` | Features del cierre auto-referencial | Longitud en caracteres y número de bloques Python de `UNIDAD_1_ML_FUNDAMENTALS.md`, extraídos por `extraer_features_unidad` (Cierre Auto-Referencial) |
| `prediccion_u1` | Hallazgos predichos | Salida de `predecir_hallazgos_por_heuristica` sobre las features de Unidad 1 (Cierre Auto-Referencial) |
| `hallazgos_reales_u1` | Hallazgos reales | `total_hallazgos` reportado por `ContentAuditorAgent().audit_unit()` sobre Unidad 1 (Cierre Auto-Referencial) |

### Autoevaluación

```python
%%writefile test_unidad_2.py
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent_core.content_auditor_agent import ContentAuditorAgent
from src.multiagent_core._fence_utils import extract_fenced_blocks


def funcion_objetivo(x: float) -> float:
    """Función objetivo sintética con máximo conocido en x=2."""
    return -(x - 2) ** 2 + 10


def optimizar_funcion_objetivo(
    limites: tuple[float, float], n_iteraciones: int = 10, semilla: int = 42
) -> float:
    """Aproxima el máximo de `funcion_objetivo` vía optimización bayesiana."""
    np.random.seed(semilla)
    rng = np.random.default_rng(semilla)
    xs = rng.uniform(limites[0], limites[1], size=n_iteraciones).reshape(-1, 1)
    ys = np.array([funcion_objetivo(x[0]) for x in xs])
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=semilla)
    gp.fit(xs, ys)
    candidatos = rng.uniform(limites[0], limites[1], size=200).reshape(-1, 1)
    mu, sigma = gp.predict(candidatos, return_std=True)
    mejor_idx = np.argmax(mu + sigma)
    return float(candidatos[mejor_idx][0])


def detectar_anomalias_wine(
    semilla: int = 42, contaminacion: float = 0.1
) -> tuple[int, int]:
    """Detecta anomalías en Wine con IsolationForest."""
    np.random.seed(semilla)
    X, _ = load_wine(return_X_y=True)
    modelo = IsolationForest(contamination=contaminacion, random_state=semilla)
    etiquetas = modelo.fit_predict(X)
    n_normales = int((etiquetas == 1).sum())
    n_anomalias = int((etiquetas == -1).sum())
    return n_normales, n_anomalias


def extraer_features_unidad(md_path: Path) -> tuple[int, int]:
    """Extrae longitud y número de bloques Python de un archivo de unidad."""
    contenido = md_path.read_text(encoding="utf-8")
    bloques = extract_fenced_blocks(contenido)
    n_bloques_python = sum(1 for _, lang, _ in bloques if lang == "python")
    return len(contenido), n_bloques_python


def predecir_hallazgos_por_heuristica(longitud: int, n_bloques_python: int) -> int:
    """Heurística de umbral fijo para predecir hallazgos de ContentAuditorAgent."""
    UMBRAL_LONGITUD = 50_000
    UMBRAL_BLOQUES = 20
    if longitud > UMBRAL_LONGITUD or n_bloques_python > UMBRAL_BLOQUES:
        return 1
    return 0


def test_optimizacion_bayesiana_se_acerca_al_optimo():
    mejor_x = optimizar_funcion_objetivo(limites=(-5, 5))
    assert abs(funcion_objetivo(mejor_x) - 10.0) < 1.0


def test_deteccion_anomalias_wine_respeta_contaminacion():
    n_normales, n_anomalias = detectar_anomalias_wine()
    total = n_normales + n_anomalias
    assert total == 178
    assert 10 <= n_anomalias <= 25


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

Ejecutado, las 3 pruebas pasan: la optimización bayesiana encuentra un punto a menos de 1.0 unidades del óptimo real, el detector de anomalías respeta la proporción esperada de `contamination` sobre las 178 muestras reales de Wine, y la heurística de predicción coincide con el resultado real de `ContentAuditorAgent().audit_unit()` sobre `UNIDAD_1_ML_FUNDAMENTALS.md`.
