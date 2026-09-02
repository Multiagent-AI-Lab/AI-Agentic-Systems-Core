# UNIDAD 1: ML Fundamentals

**Curso:** AI-Agentic-Systems-Core — UCEMICH

## Prefacio

La Unidad 0 dio el lenguaje matemático formal (álgebra lineal, cálculo, probabilidad, información) que sostiene cualquier sistema de IA. Esta unidad da el primer paso de ingeniería: define qué es un **agente** como pieza de software, y muestra el ciclo de vida completo de dos modelos de Machine Learning —uno clásico, uno neuronal— construidos con el mismo rigor de verificación que exigirá el resto del curso. El Diccionario de Variables de la Unidad 0 no se repite aquí; los símbolos nuevos de esta unidad son propios del contexto de ML (datasets, hiperparámetros, métricas).

---

## ¿Qué es un Agente? — Anatomía de un Agente

**Agente = Modelo + Harness**

El **Modelo** es el componente que razona o predice: puede ser un árbol de decisión, una red neuronal, o un LLM — cualquier función entrenada o programada que, dada una entrada, produce una salida. El **Harness** es todo lo que rodea a ese modelo para convertirlo en un agente capaz de operar en el mundo real. El Harness provee tres piezas:

- **Tools**: APIs, ejecutores de código, navegadores — lo que permite al agente actuar sobre su entorno en vez de solo predecir.
- **Memory**: historial de conversación, ventana de contexto, estado persistente — lo que permite al agente recordar interacciones pasadas.
- **Guardrails**: reglas de seguridad, límites de permisos, verificaciones que atrapan errores — lo que impide que el agente ejecute acciones no autorizadas o dañinas.

Un modelo sin Harness es solo una función matemática aislada: un árbol de decisión que predice la especie de una flor no es, por sí mismo, un agente. Se vuelve agente cuando se le da la capacidad de **actuar** (Tools), **recordar** (Memory) y **actuar dentro de límites** (Guardrails).

### Tools: el agente puede actuar

Una Tool es, en su forma más simple, una función que el agente puede invocar por nombre. El siguiente bloque construye un registro mínimo de Tools —un diccionario que mapea nombres a funciones— y muestra cómo el agente "decide" cuál invocar:

```python
from typing import Callable


def sumar(a: float, b: float) -> float:
    """Suma dos numeros. Tool minima que el agente puede invocar."""
    return a + b


def multiplicar(a: float, b: float) -> float:
    """Multiplica dos numeros. Tool minima que el agente puede invocar."""
    return a * b


herramientas: dict[str, Callable[[float, float], float]] = {
    "sumar": sumar,
    "multiplicar": multiplicar,
}

resultado = herramientas["sumar"](3.0, 4.0)
print(f"Resultado de invocar la tool sumar: {resultado}")
```

Ejecutado, imprime `Resultado de invocar la tool sumar: 7.0`. El patrón —un diccionario de nombre → función invocable— es exactamente el mecanismo que usan los frameworks de agentes reales (LangChain, `google.genai` function calling) para exponer capacidades a un LLM: el modelo elige un nombre de tool, el harness lo traduce a una llamada de función real.

### Memory: el agente puede recordar

La memoria de un agente rara vez es ilimitada — igual que la ventana de contexto de un LLM, tiene un tamaño máximo. El siguiente bloque implementa una memoria de ventana deslizante que retiene solo los últimos `tamano_maximo` mensajes:

```python
from collections import deque


class MemoriaConVentana:
    """Memoria de conversacion con ventana deslizante de tamano fijo.

    Simula la ventana de contexto de un agente: solo retiene los
    ultimos `tamano_maximo` mensajes, descartando los mas antiguos.
    """

    def __init__(self, tamano_maximo: int = 3) -> None:
        """Inicializa la memoria con una ventana de tamaño `tamano_maximo`."""
        self.tamano_maximo = tamano_maximo
        self.historial: deque[str] = deque(maxlen=tamano_maximo)

    def agregar(self, mensaje: str) -> None:
        """Agrega un mensaje, descartando el mas antiguo si excede el limite."""
        self.historial.append(mensaje)


memoria = MemoriaConVentana(tamano_maximo=3)
for mensaje in ["Hola", "Como estas", "Bien gracias", "Que necesitas"]:
    memoria.agregar(mensaje)

print(f"Historial retenido: {list(memoria.historial)}")
assert len(memoria.historial) == 3
assert list(memoria.historial) == ["Como estas", "Bien gracias", "Que necesitas"]
```

Ejecutado, confirma que de los 4 mensajes agregados solo sobreviven los últimos 3 — el primero (`"Hola"`) se descarta automáticamente por el `maxlen` de `deque`, tal como un LLM "olvida" los turnos más antiguos de una conversación cuando excede su ventana de contexto.

### Guardrails: el agente actúa dentro de límites

Un guardrail es una verificación explícita que se ejecuta **antes** de que el agente actúe, y que puede bloquear la acción. El siguiente bloque implementa una lista blanca de acciones permitidas:

```python
def validar_accion_permitida(accion: str, acciones_permitidas: set[str]) -> bool:
    """Guardrail minimo: rechaza cualquier accion fuera de la lista blanca.

    Args:
        accion: Nombre de la accion que el agente quiere ejecutar.
        acciones_permitidas: Conjunto de acciones autorizadas explicitamente.

    Returns:
        True si la accion esta permitida, False si debe bloquearse.
    """
    return accion in acciones_permitidas


acciones_permitidas = {"leer_archivo", "sumar", "multiplicar"}
puede_leer = validar_accion_permitida("leer_archivo", acciones_permitidas)
puede_borrar = validar_accion_permitida("borrar_archivo", acciones_permitidas)
print(f"¿Puede leer_archivo? {puede_leer}")
print(f"¿Puede borrar_archivo? {puede_borrar}")
assert puede_leer is True
assert puede_borrar is False
```

Ejecutado, confirma que `"leer_archivo"` está en la lista blanca (`True`) y `"borrar_archivo"` no lo está (`False`) — el mismo principio que usan los sistemas de permisos de agentes de producción (Claude Code incluido): nunca ejecutar una acción que no fue autorizada explícitamente.

Con Tools, Memory y Guardrails definidos, el resto de esta unidad construye dos **Modelos** concretos —un árbol de decisión y una red neuronal— y cierra mostrando cómo esos mismos tres conceptos de Harness aparecen en un agente real del propio repositorio del curso.

---

## ML Clásico: Clasificación con Iris

El dataset Iris (Fisher, 1936) es el ejemplo canónico de clasificación supervisada: 150 flores de 3 especies, descritas por 4 medidas (largo/ancho de sépalo y pétalo). Un árbol de decisión es el modelo más simple e interpretable para este problema: aprende una secuencia de reglas del tipo "si el ancho del pétalo es menor a X, entonces...".

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def entrenar_clasificador_iris(semilla: int = 42) -> float:
    """Entrena un árbol de decisión sobre el dataset Iris y retorna su exactitud.

    Args:
        semilla: Semilla aleatoria para reproducibilidad del split train/test.

    Returns:
        Exactitud (accuracy) del modelo sobre el conjunto de prueba.
    """
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    modelo = DecisionTreeClassifier(random_state=semilla)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    return accuracy_score(y_test, predicciones)


exactitud = entrenar_clasificador_iris()
print(f"Exactitud: {exactitud:.2%}")
```

Ejecutado, este bloque imprime `Exactitud: 100.00%`. Iris es un dataset linealmente muy separable entre especies (en particular, la especie *setosa* se distingue trivialmente de las otras dos), por lo que un árbol de decisión sin restricciones de profundidad lo resuelve sin error en el conjunto de prueba — un resultado real, no inflado, pero que también sirve de advertencia pedagógica: una exactitud perfecta en un dataset pequeño y fácil no garantiza que el modelo generalice igual de bien a datos más difíciles (ver "Iteración" más abajo).

---

## Redes Neuronales: Regresión con California Housing

El dataset California Housing (20,640 distritos censales, 8 features socioeconómicas) es un problema de **regresión**: predecir el valor medio de vivienda, un número continuo, no una categoría. A diferencia de Iris, aquí se usa un `MLPRegressor` (perceptrón multicapa) de scikit-learn: una red neuronal densa con capas ocultas.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def entrenar_red_california_housing(semilla: int = 42) -> tuple[float, float]:
    """Entrena una red neuronal (MLP) sobre California Housing.

    Args:
        semilla: Semilla aleatoria para reproducibilidad del split y los
            pesos iniciales de la red.

    Returns:
        Tupla (RMSE, R2) del modelo evaluado sobre el conjunto de prueba.
    """
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    modelo = MLPRegressor(
        hidden_layer_sizes=(32, 16), max_iter=500, random_state=semilla
    )
    modelo.fit(X_train_esc, y_train)
    predicciones = modelo.predict(X_test_esc)
    rmse = mean_squared_error(y_test, predicciones) ** 0.5
    r2 = r2_score(y_test, predicciones)
    return rmse, r2


rmse, r2 = entrenar_red_california_housing()
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
```

Ejecutado, produce `RMSE: 0.5393` y `R2: 0.7781` (valores en unidades de cientos de miles de dólares, la escala nativa del dataset). A diferencia de Iris, aquí es indispensable **escalar** las features con `StandardScaler` antes de entrenar: una red neuronal converge mal (o no converge) cuando sus entradas tienen escalas muy distintas entre sí, mientras que un árbol de decisión —que solo compara umbrales feature por feature— es indiferente a la escala.

---

## 🔄 El Ciclo del Agente

Las dos secciones anteriores entrenaron dos modelos sin discutir *por qué* se eligió cada arquitectura, ni qué pasaría después de entrenarlos. Esta sección cierra la unidad aplicando el ciclo de vida completo de ingeniería a esa decisión.

### Selección de Arquitectura

¿Árbol de decisión o red neuronal para cada uno de estos dos problemas? La respuesta depende de la estructura del problema, no de preferencia:

- **Iris (clasificación, 4 features, 150 muestras)**: un árbol de decisión es la elección correcta. El dataset es pequeño, las fronteras entre clases son casi lineales, y un árbol da además **interpretabilidad total** — se puede leer la secuencia exacta de reglas que llevó a cada predicción. Una red neuronal aquí sería sobre-ingeniería: más parámetros que muestras de entrenamiento, mayor riesgo de sobreajuste, y una caja negra donde no hacía falta una.
- **California Housing (regresión, 8 features, 20,640 muestras)**: hay suficientes datos para que una red neuronal aprenda interacciones no lineales entre features (por ejemplo, cómo el ingreso medio interactúa con la ubicación geográfica) que un árbol simple aproximaría de forma más tosca. El costo es interpretabilidad: los pesos de un MLP no se leen como reglas humanas.

La regla general: la complejidad del modelo debe ser proporcional a la complejidad de la señal y a la cantidad de datos disponibles — no a la sofisticación percibida del método.

### Diseño

**Context Engineering** (qué información entra al modelo): en Iris, las 4 features (largo/ancho de sépalo y pétalo) son exactamente las medidas relevantes — no hay features irrelevantes que filtrar. En California Housing, las 8 features (ingreso medio, edad de la vivienda, habitaciones, ubicación, etc.) requieren una decisión de diseño adicional: **escalarlas** antes de entrenar, porque conviven en escalas muy distintas (ingreso en decenas de miles, latitud en grados). Decidir qué entra al modelo — y en qué forma — es la primera decisión de diseño de cualquier sistema de ML, y es análoga a decidir qué contexto recibe un LLM en un prompt.

**Spec-Driven Development**: antes de entrenar, conviene escribir explícitamente qué debe lograr el modelo. Para el clasificador de Iris: *"dado un vector de 4 medidas florales, predecir una de 3 especies con exactitud superior al 80% sobre datos no vistos"*. Esa especificación (el `assert exactitud > 0.8` de la sección de Autoevaluación) existe *antes* de que el modelo se considere terminado — no se ajusta después para que el modelo entrenado "pase".

### Implementación

El árbol de decisión y la red neuronal de esta unidad se ajustan llamando a `.fit()` — sus parámetros internos (umbrales de decisión, pesos de la red) se optimizan automáticamente contra los datos de entrenamiento. Si el "modelo" fuera en cambio un LLM (como en la Unidad 3), no habría pesos que reentrenar: la pieza que se optimiza es el **prompt** — las instrucciones y ejemplos que guían al modelo congelado. **DSPy** es el framework que la Unidad 3 introduce para esa tarea: en vez de ajustar un prompt a mano por prueba y error, DSPy trata el prompt como un parámetro optimizable y lo ajusta automáticamente contra ejemplos etiquetados, de la misma forma en que `DecisionTreeClassifier.fit()` ajusta los umbrales del árbol contra `X_train, y_train`. La diferencia es qué se optimiza (pesos vs. texto de instrucciones), no la lógica del proceso.

### Evaluación

Cada tipo de problema exige una métrica distinta:

- **Clasificación (Iris)**: `accuracy_score` — la fracción de predicciones correctas — es la métrica usada arriba. Para un análisis más fino (qué especies se confunden entre sí), la herramienta estándar es la **matriz de confusión** (`sklearn.metrics.confusion_matrix`), que esta unidad no ejecuta por no ser indispensable con exactitud del 100%, pero que sería la primera herramienta a inspeccionar si la exactitud fuera menor.
- **Regresión (California Housing)**: `accuracy_score` no aplica — no hay "clases correctas o incorrectas" al predecir un número continuo. Por eso se usan RMSE (error cuadrático medio, en las unidades originales del precio) y R² (fracción de la varianza explicada por el modelo, entre 0 y 1). Un R² de 0.78 significa que el modelo explica el 78% de la variación en los precios de vivienda del conjunto de prueba.

### Despliegue

El costo de servir estos dos modelos en producción es radicalmente distinto. Un árbol de decisión evalúa una predicción recorriendo una secuencia de comparaciones (`if feature <= umbral`) — complejidad $O(\text{profundidad del árbol})$, típicamente microsegundos, sin necesidad de GPU. Una red neuronal, incluso una tan pequeña como el MLP de 2 capas ocultas de esta unidad, requiere multiplicaciones de matrices por cada predicción — más costoso por muestra, aunque sigue siendo trivial en CPU a esta escala. La brecha se vuelve crítica al comparar contra un LLM: un árbol de decisión responde en microsegundos sin costo de API, mientras que invocar un LLM para la misma tarea implica latencia de red, costo por token, y una GPU (propia o rentada) del lado del proveedor. Esta es la razón práctica por la que un agente bien diseñado no usa un LLM para tareas que un modelo clásico resuelve igual de bien y a una fracción del costo — la Unidad 3 retoma esta disyuntiva explícitamente al decidir qué componentes de un agente usan un LLM y cuáles no.

### Iteración

Si la exactitud del clasificador de Iris hubiera sido baja (por ejemplo, 60%), los siguientes pasos —en orden de costo creciente— serían: (1) revisar el Diseño, verificando que las features de entrada sean las correctas y estén bien escaladas; (2) ajustar hiperparámetros del árbol (profundidad máxima, criterio de división); (3) recolectar más datos etiquetados; (4) solo al final, considerar un modelo más complejo. Para la red de California Housing, un R² bajo llevaría primero a probar más neuronas o capas (`hidden_layer_sizes`), y solo después a reconsiderar la arquitectura completa. La exactitud perfecta obtenida en Iris en esta unidad es, en sí misma, una señal a vigilar: en un problema real con datos más ruidosos, una exactitud del 100% en el conjunto de prueba suele ser síntoma de una fuga de datos (*data leakage*) o de un conjunto de prueba demasiado fácil — no de un modelo perfecto. Iterar también significa cuestionar los resultados que se ven "demasiado buenos".

---

## Cierre Auto-Referencial: Clasificando Hallazgos de CodeAuditorAgent

Esta unidad definió Tools, Memory y Guardrails como las tres piezas del Harness de un agente. `CodeAuditorAgent` — un agente real de este mismo repositorio (`src/multiagent_core/code_auditor_agent.py`) — es un ejemplo concreto de Tool: un componente invocable que analiza código Python y reporta hallazgos de estilo y seguridad. El siguiente bloque lo importa y usa de verdad (no lo simula) para construir un pequeño clasificador que decide qué tan riesgoso es un fragmento de código según la cantidad de hallazgos que el auditor reporta:

```python
from src.multiagent_core.code_auditor_agent import CodeAuditorAgent


def clasificar_severidad_por_cantidad_de_hallazgos(codigo: str) -> str:
    """Clasifica código en 'limpio', 'con advertencias' o 'riesgoso' según
    la cantidad de hallazgos que reporta CodeAuditorAgent.

    Args:
        codigo: Código fuente Python a clasificar.

    Returns:
        Una de 'limpio', 'con advertencias', 'riesgoso'.
    """
    auditor = CodeAuditorAgent()
    total = len(auditor.audit_style(codigo)) + len(auditor.audit_security(codigo))
    if total == 0:
        return "limpio"
    if total <= 2:
        return "con advertencias"
    return "riesgoso"


codigo_limpio = "tasa_aprendizaje = 0.01\nnum_epocas = 100"
codigo_con_advertencias = "tasaAprendizaje = 0.01\nnumEpocas = 100"
codigo_riesgoso = (
    'tasaAprendizaje = 0.01\n'
    'numEpocas = 100\n'
    'eval("print(1)")\n'
    'exec("print(2)")'
)

for nombre, codigo in [
    ("limpio", codigo_limpio),
    ("con advertencias", codigo_con_advertencias),
    ("riesgoso", codigo_riesgoso),
]:
    severidad = clasificar_severidad_por_cantidad_de_hallazgos(codigo)
    print(f"Código {nombre!r} clasificado como: {severidad}")
```

Ejecutado, imprime `Código 'limpio' clasificado como: limpio`, `Código 'con advertencias' clasificado como: con advertencias` y `Código 'riesgoso' clasificado como: riesgoso` — el auditor real detecta 0 hallazgos en el primer fragmento (snake_case correcto, sin riesgos), 2 hallazgos de estilo CamelCase en el segundo, y 2 hallazgos de estilo CamelCase más 2 de seguridad (`eval()` y `exec()`) en el tercero. Este es el mismo principio de Guardrails de la sección de Anatomía del Agente, aplicado por un agente real del curso: una verificación automática, ejecutada antes de aceptar código, que clasifica el riesgo en vez de solo reportarlo en bruto.

### Diccionario de Variables

| Símbolo | Nombre | Descripción |
|---|---|---|
| `herramientas` | Registro de Tools | Diccionario nombre → función invocable por el agente (sección Anatomía) |
| `resultado` | Salida de una Tool | Resultado de invocar `herramientas["sumar"]` (sección Anatomía) |
| `memoria` | Instancia de memoria con ventana | `MemoriaConVentana` con `tamano_maximo=3` (sección Anatomía) |
| `tamano_maximo` | Tamaño máximo de la ventana de memoria | Límite de mensajes retenidos antes de descartar los más antiguos (sección Anatomía) |
| `acciones_permitidas` | Lista blanca de Guardrail | Conjunto de acciones que el agente puede ejecutar sin bloqueo (sección Anatomía) |
| `X`, `y` | Features y etiquetas/objetivo | Entradas y salida de `load_iris`/`fetch_california_housing` (ML Clásico, Redes Neuronales) |
| `semilla` | Semilla aleatoria | Fija el split train/test y la inicialización de pesos, para reproducibilidad (ambas secciones de ML) |
| `modelo` | Estimador de scikit-learn | `DecisionTreeClassifier` o `MLPRegressor` según la sección |
| `exactitud` | Accuracy del clasificador | Fracción de predicciones correctas de `entrenar_clasificador_iris` (ML Clásico) |
| `escalador` | `StandardScaler` ajustado | Normaliza las features de California Housing antes de entrenar la red (Redes Neuronales) |
| `rmse`, `r2` | Métricas de regresión | Error cuadrático medio (raíz) y coeficiente de determinación de la red (Redes Neuronales) |
| `auditor` | Instancia de `CodeAuditorAgent` | Agente real usado para clasificar severidad de código (Cierre Auto-Referencial) |
| `total` | Conteo de hallazgos | Suma de hallazgos de estilo y seguridad reportados por `auditor` (Cierre Auto-Referencial) |

### Autoevaluación

```python
%%writefile test_unidad_1.py
import sys
from pathlib import Path

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent_core.code_auditor_agent import CodeAuditorAgent


def entrenar_clasificador_iris(semilla: int = 42) -> float:
    """Entrena un árbol de decisión sobre Iris y retorna su exactitud."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    modelo = DecisionTreeClassifier(random_state=semilla)
    modelo.fit(X_train, y_train)
    return accuracy_score(y_test, modelo.predict(X_test))


def entrenar_red_california_housing(semilla: int = 42) -> tuple[float, float]:
    """Entrena un MLPRegressor sobre California Housing y retorna (RMSE, R2)."""
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)
    modelo = MLPRegressor(
        hidden_layer_sizes=(32, 16), max_iter=500, random_state=semilla
    )
    modelo.fit(X_train_esc, y_train)
    predicciones = modelo.predict(X_test_esc)
    rmse = mean_squared_error(y_test, predicciones) ** 0.5
    r2 = r2_score(y_test, predicciones)
    return rmse, r2


def clasificar_severidad_por_cantidad_de_hallazgos(codigo: str) -> str:
    """Clasifica código según la cantidad de hallazgos de CodeAuditorAgent."""
    auditor = CodeAuditorAgent()
    total = len(auditor.audit_style(codigo)) + len(auditor.audit_security(codigo))
    if total == 0:
        return "limpio"
    if total <= 2:
        return "con advertencias"
    return "riesgoso"


def test_clasificador_iris_supera_umbral_minimo():
    assert entrenar_clasificador_iris() > 0.8


def test_red_california_housing_supera_r2_minimo():
    _, r2 = entrenar_red_california_housing()
    assert r2 > 0.5


def test_clasificar_severidad_codigo_limpio():
    codigo = "tasa_aprendizaje = 0.01"
    assert clasificar_severidad_por_cantidad_de_hallazgos(codigo) == "limpio"


def test_clasificar_severidad_codigo_riesgoso():
    codigo = (
        'tasaAprendizaje = 0.01\n'
        'numEpocas = 100\n'
        'eval("print(1)")\n'
        'exec("print(2)")'
    )
    assert clasificar_severidad_por_cantidad_de_hallazgos(codigo) == "riesgoso"
```

```python
!pytest test_unidad_1.py -v
```

Ejecutado, las 4 pruebas pasan: el árbol de decisión supera 80% de exactitud, la red neuronal supera 0.5 de R², y el clasificador de severidad distingue correctamente código limpio de código riesgoso usando el agente real del repositorio.
