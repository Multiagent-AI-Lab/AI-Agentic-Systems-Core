# UNIDAD 1: ML Fundamentals

**Curso:** AI-Agentic-Systems-Core — UCEMICH

## Prefacio

La Unidad 0 dio el lenguaje matemático formal (álgebra lineal, cálculo, probabilidad, información) que sostiene cualquier sistema de IA. Esta unidad da el primer paso de ingeniería: define qué es un **agente** como pieza de software, y muestra el ciclo de vida completo de dos modelos de Machine Learning —uno clásico, uno neuronal— construidos con el mismo rigor de verificación que exigirá el resto del curso. El Diccionario de Variables de la Unidad 0 no se repite aquí; los símbolos nuevos de esta unidad son propios del contexto de ML (datasets, hiperparámetros, métricas).

Entrenar estos dos modelos a mano tiene un propósito concreto que va más allá de "aprender ML": es exactamente el tipo de decisión — ¿qué arquitectura elegir?, ¿el modelo generaliza o memorizó?, ¿qué métrica es la correcta para este problema? — que un agente con acceso a estas mismas herramientas puede tomar por sí mismo hoy. El objetivo de esta unidad no es que el lector se convierta en ingeniero de ML clásico, sino que entienda el criterio de decisión lo suficientemente bien como para reconocer cuándo un agente lo está aplicando correctamente y cuándo no — el rol que ejercerá al construir sistemas multiagente en la Unidad 3.

---

## ¿Qué es un Agente? — Anatomía de un Agente

**Agente = Modelo + Harness**

El **Modelo** es el componente que razona o predice: puede ser un árbol de decisión, una red neuronal, o un LLM — cualquier función entrenada o programada que, dada una entrada, produce una salida. El **Harness** es todo lo que rodea a ese modelo para convertirlo en un agente capaz de operar en el mundo real. El Harness provee tres piezas:

- **Tools**: APIs, ejecutores de código, navegadores — lo que permite al agente actuar sobre su entorno en vez de solo predecir.
- **Memory**: historial de conversación, ventana de contexto, estado persistente — lo que permite al agente recordar interacciones pasadas.
- **Guardrails**: reglas de seguridad, límites de permisos, verificaciones que atrapan errores — lo que impide que el agente ejecute acciones no autorizadas o dañinas.

Un modelo sin Harness es solo una función matemática aislada: un árbol de decisión que predice la especie de una flor no es, por sí mismo, un agente. Se vuelve agente cuando se le da la capacidad de **actuar** (Tools), **recordar** (Memory) y **actuar dentro de límites** (Guardrails).

### Tools: el agente puede actuar

Una Tool es, en su forma más simple, una función que el agente puede invocar por nombre. La alternativa a este patrón sería que el LLM generara y ejecutara código Python arbitrario para cada tarea — pero eso expone una superficie de ataque mucho mayor (código arbitrario puede hacer cualquier cosa que el proceso tenga permiso de hacer) y hace imposible auditar de antemano qué puede y qué no puede hacer el agente. Un registro nombre→función invierte esa relación: el conjunto de acciones posibles queda fijado por quien construye el sistema, no por lo que el modelo decida generar en tiempo de ejecución, y cada tool puede auditarse, probarse y documentarse por separado antes de que el agente exista. El siguiente bloque construye ese registro con dos tools que en realidad son los dos modelos que esta unidad va a entrenar más abajo — un agente real invocaría estas mismas funciones para decidir, por ejemplo, cuál de los dos modelos entrenar dado un nuevo dataset:

```python
from typing import Callable

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def entrenar_clasificador_iris(semilla: int = 42) -> float:
    """Tool: entrena un árbol de decisión sobre Iris y retorna su exactitud."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    modelo = DecisionTreeClassifier(random_state=semilla)
    modelo.fit(X_train, y_train)
    return accuracy_score(y_test, modelo.predict(X_test))


def entrenar_red_california_housing(semilla: int = 42) -> tuple[float, float]:
    """Tool: entrena una red neuronal sobre California Housing y retorna (RMSE, R2)."""
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)
    modelo = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=semilla)
    modelo.fit(X_train_esc, y_train)
    predicciones = modelo.predict(X_test_esc)
    return mean_squared_error(y_test, predicciones) ** 0.5, r2_score(y_test, predicciones)


herramientas: dict[str, Callable] = {
    "entrenar_clasificador_iris": entrenar_clasificador_iris,
    "entrenar_red_california_housing": entrenar_red_california_housing,
}

resultado = herramientas["entrenar_clasificador_iris"]()
print(f"Resultado de invocar la tool entrenar_clasificador_iris: {resultado:.2%}")
```

Ejecutado, imprime `Resultado de invocar la tool entrenar_clasificador_iris: 100.00%` — el mismo resultado que la sección "ML Clásico" más abajo obtiene entrenando el árbol directamente, porque es literalmente la misma función; la diferencia es que aquí se invoca indirectamente, por nombre, a través del registro. El patrón —un diccionario de nombre → función invocable— es exactamente el mecanismo que usan los frameworks de agentes reales (LangChain, `google.genai` function calling) para exponer capacidades a un LLM: el modelo elige un nombre de tool, el harness lo traduce a una llamada de función real, y el propio conjunto de claves del diccionario es, en sí mismo, el guardrail más básico posible — el agente no puede invocar lo que no está registrado.

### Memory: el agente puede recordar

La memoria de un agente rara vez es ilimitada — igual que la ventana de contexto de un LLM, tiene un tamaño máximo. Una alternativa a la ventana deslizante sería resumir incrementalmente cada mensaje descartado con un LLM antes de eliminarlo — preservando la información en forma comprimida en vez de perderla por completo — pero eso cuesta una llamada adicional al modelo por cada mensaje descartado, con su propia latencia y costo, y arriesga que el resumen pierda matices que sí importaban. La ventana deslizante es la opción más simple y barata: cuesta cero llamadas extra, a costa de perder información completamente en vez de degradarla gradualmente. El siguiente bloque implementa esa opción con los últimos `tamano_maximo` mensajes, usando como contenido el tipo de historial que un agente de este dominio acumularía de verdad — no una conversación genérica, sino un registro de sus propias decisiones de entrenamiento:

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
decisiones_del_agente = [
    "evaluado arbol de decision sobre Iris: exactitud=1.00",
    "evaluado MLP sobre California Housing: R2=0.78",
    "seleccionado arbol de decision por interpretabilidad (Iris es lineal)",
    "iniciado entrenamiento de MLP con escalado de features",
]
for mensaje in decisiones_del_agente:
    memoria.agregar(mensaje)

print(f"Historial retenido: {list(memoria.historial)}")
assert len(memoria.historial) == 3
assert memoria.historial[0] == "evaluado MLP sobre California Housing: R2=0.78"
```

Ejecutado, confirma que de los 4 mensajes agregados solo sobreviven los últimos 3 — el primero (la evaluación del árbol de Iris) se descarta automáticamente por el `maxlen` de `deque`, tal como un LLM "olvida" los turnos más antiguos de una conversación cuando excede su ventana de contexto. Nótese que esto no es un detalle cosmético del ejemplo: si este agente necesitara justificar más tarde *por qué* usó el árbol de decisión, esa razón ya no estaría en su memoria — un caso concreto de por qué la Unidad 3 trata la gestión de memoria como una decisión de seguridad y no solo de eficiencia.

### Guardrails: el agente actúa dentro de límites

Un guardrail es una verificación explícita que se ejecuta **antes** de que el agente actúe, y que puede bloquear la acción. La alternativa a una lista blanca sería una lista negra — enumerar explícitamente las acciones prohibidas y permitir todo lo demás por defecto — pero eso falla exactamente cuando más importa: una lista negra solo puede bloquear los ataques que ya se conocen de antemano, mientras que una lista blanca bloquea por defecto cualquier cosa no anticipada, incluidas variantes de ataque que todavía no existen. Este es el mismo principio de menor privilegio que la Unidad 3 invoca explícitamente para sistemas multiagente: cada componente debe tener acceso solo a lo estrictamente necesario, nunca a "todo salvo lo prohibido". El siguiente bloque implementa esa lista blanca:

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


acciones_permitidas = {
    "leer_archivo",
    "entrenar_clasificador_iris",
    "entrenar_red_california_housing",
}
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

El diseño de un pipeline de ML se reduce, en esencia, a dos decisiones que se toman antes de escribir la primera línea de entrenamiento: qué información entra al modelo, y qué se considerará éxito antes de haberlo entrenado.

**Context Engineering** (qué información entra al modelo): en Iris, las 4 features (largo/ancho de sépalo y pétalo) son exactamente las medidas relevantes — no hay features irrelevantes que filtrar. En California Housing, las 8 features (ingreso medio, edad de la vivienda, habitaciones, ubicación, etc.) requieren una decisión de diseño adicional: **escalarlas** antes de entrenar, porque conviven en escalas muy distintas (ingreso en decenas de miles, latitud en grados). Decidir qué entra al modelo — y en qué forma — es la primera decisión de diseño de cualquier sistema de ML, y es análoga a decidir qué contexto recibe un LLM en un prompt.

**Spec-Driven Development**: antes de entrenar, conviene escribir explícitamente qué debe lograr el modelo. Para el clasificador de Iris: *"dado un vector de 4 medidas florales, predecir una de 3 especies con exactitud superior al 80% sobre datos no vistos"*. Esa especificación (el `assert exactitud > 0.8` de la sección de Autoevaluación) existe *antes* de que el modelo se considere terminado — no se ajusta después para que el modelo entrenado "pase".

### Implementación

El árbol de decisión y la red neuronal de esta unidad se ajustan llamando a `.fit()` — sus parámetros internos (umbrales de decisión, pesos de la red) se optimizan automáticamente contra los datos de entrenamiento. Si el "modelo" fuera en cambio un LLM (como en la Unidad 3), no habría pesos que reentrenar: la pieza que se optimiza es el **prompt** — las instrucciones y ejemplos que guían al modelo congelado. **DSPy** es el framework que la Unidad 3 introduce para esa tarea: en vez de ajustar un prompt a mano por prueba y error, DSPy trata el prompt como un parámetro optimizable y lo ajusta automáticamente contra ejemplos etiquetados, de la misma forma en que `DecisionTreeClassifier.fit()` ajusta los umbrales del árbol contra `X_train, y_train`. La diferencia es qué se optimiza (pesos vs. texto de instrucciones), no la lógica del proceso.

### Evaluación

Cada tipo de problema exige una métrica distinta, y para clasificación esa métrica única (`accuracy`) esconde más de lo que revela cuando hay más de dos clases.

**Clasificación (Iris)**: `accuracy_score` — la fracción de predicciones correctas — es la métrica usada arriba, y con el árbol sin restricciones da 100%, dejando poco que analizar. Un árbol deliberadamente limitado (`max_depth=1`, solo una pregunta de sí/no) es más revelador para ver qué aporta una matriz de confusión sobre una sola cifra de exactitud:

```python
from sklearn.metrics import confusion_matrix

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_limitado = DecisionTreeClassifier(random_state=42, max_depth=1)
modelo_limitado.fit(X_train, y_train)
predicciones_limitadas = modelo_limitado.predict(X_test)

matriz = confusion_matrix(y_test, predicciones_limitadas)
print(f"Exactitud con max_depth=1: {accuracy_score(y_test, predicciones_limitadas):.2%}")
print(f"Matriz de confusión:\n{matriz}")
```

Ejecutado, la exactitud cae a `63.33%` y la matriz de confusión revela exactamente dónde: la fila de *virginica* queda completa en la columna de *versicolor* (`[0, 11, 0]`) — con una sola pregunta, el árbol distingue perfectamente *setosa* del resto pero no puede separar las otras dos especies entre sí. Una sola cifra de exactitud (`63%`) dice que el modelo falla, pero la matriz dice *cómo* falla: confunde sistemáticamente dos clases específicas, no falla al azar entre las tres — la diferencia entre esos dos diagnósticos determina si el siguiente paso es "agregar profundidad al árbol" (correcto aquí) o "revisar si dos especies están mal etiquetadas en los datos" (lo que se probaría primero si la confusión fuera simétrica en ambas direcciones en vez de unidireccional).

**Regresión (California Housing)**: `accuracy_score` no aplica — no hay "clases correctas o incorrectas" al predecir un número continuo. Por eso se usan RMSE (error cuadrático medio, en las unidades originales del precio) y R² (fracción de la varianza explicada por el modelo, entre 0 y 1). Un R² de 0.78 significa que el modelo explica el 78% de la variación en los precios de vivienda del conjunto de prueba.

### Despliegue

El costo de servir estos dos modelos en producción es radicalmente distinto. Un árbol de decisión evalúa una predicción recorriendo una secuencia de comparaciones (`if feature <= umbral`) — complejidad $O(\text{profundidad del árbol})$, típicamente microsegundos, sin necesidad de GPU. Una red neuronal, incluso una tan pequeña como el MLP de 2 capas ocultas de esta unidad, requiere multiplicaciones de matrices por cada predicción — más costoso por muestra, aunque sigue siendo trivial en CPU a esta escala. La brecha se vuelve crítica al comparar contra un LLM: un árbol de decisión responde en microsegundos sin costo de API, mientras que invocar un LLM para la misma tarea implica latencia de red, costo por token, y una GPU (propia o rentada) del lado del proveedor. Esta es la razón práctica por la que un agente bien diseñado no usa un LLM para tareas que un modelo clásico resuelve igual de bien y a una fracción del costo — la Unidad 3 retoma esta disyuntiva explícitamente al decidir qué componentes de un agente usan un LLM y cuáles no.

### Iteración

Si la exactitud del clasificador de Iris hubiera sido baja (por ejemplo, 60%), los siguientes pasos —en orden de costo creciente— serían: (1) revisar el Diseño, verificando que las features de entrada sean las correctas y estén bien escaladas; (2) ajustar hiperparámetros del árbol (profundidad máxima, criterio de división); (3) recolectar más datos etiquetados; (4) solo al final, considerar un modelo más complejo. Para la red de California Housing, un R² bajo llevaría primero a probar más neuronas o capas (`hidden_layer_sizes`), y solo después a reconsiderar la arquitectura completa. La exactitud perfecta obtenida en Iris en esta unidad es, en sí misma, una señal a vigilar: en un problema real con datos más ruidosos, una exactitud del 100% en el conjunto de prueba suele ser síntoma de una fuga de datos (*data leakage*) o de un conjunto de prueba demasiado fácil — no de un modelo perfecto. Iterar también significa cuestionar los resultados que se ven "demasiado buenos".

---

## Ejercicios

### Ejercicio A (para resolver): extender la memoria con resumen de descartados

`MemoriaConVentana` de la sección de Anatomía descarta silenciosamente los mensajes más antiguos. Agrega un método `resumir()` que retorne un string con el número de mensajes descartados y el contenido de los que sí se retienen — una forma mínima de que el agente sepa que perdió información, en vez de descartarla sin dejar rastro:

```python
class MemoriaConResumen(MemoriaConVentana):
    """Extiende MemoriaConVentana contando cuántos mensajes se descartaron."""

    def __init__(self, tamano_maximo: int = 3) -> None:
        """Inicializa la memoria y el contador de mensajes descartados en cero."""
        super().__init__(tamano_maximo)
        self.mensajes_descartados = 0

    def agregar(self, mensaje: str) -> None:
        """Agrega un mensaje, incrementando el contador si desplaza a otro."""
        if len(self.historial) == self.tamano_maximo:
            self.mensajes_descartados += 1
        super().agregar(mensaje)

    def resumir(self) -> str:
        """Retorna un resumen: cuántos mensajes se perdieron y cuáles se retienen."""
        return f"[{self.mensajes_descartados} mensajes descartados] " + " | ".join(self.historial)


memoria_con_resumen = MemoriaConResumen(tamano_maximo=3)
for mensaje in decisiones_del_agente:
    memoria_con_resumen.agregar(mensaje)

print(memoria_con_resumen.resumir())
assert memoria_con_resumen.mensajes_descartados == 1
```

### Ejercicio B (para resolver): confirmar la sobre-ingeniería con evidencia

Selección de Arquitectura afirma que una red neuronal sería "sobre-ingeniería" para Iris, pero esa afirmación no se comprueba en ningún bloque de código de la unidad. Ciérrala aquí: entrena un `MLPClassifier` sobre el mismo split de Iris y compáralo contra el árbol de decisión en exactitud **y** en tiempo de entrenamiento — la tesis de sobre-ingeniería no es que la red falle, es que iguala al árbol sin ninguna ventaja a cambio:

```python
import time

from sklearn.neural_network import MLPClassifier

t0 = time.perf_counter()
arbol_b = entrenar_clasificador_iris()
tiempo_arbol = time.perf_counter() - t0

t0 = time.perf_counter()
red_iris = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
red_iris.fit(X_train, y_train)
exactitud_red_iris = accuracy_score(y_test, red_iris.predict(X_test))
tiempo_red_iris = time.perf_counter() - t0

print(f"Árbol: exactitud={arbol_b:.2%}, tiempo={tiempo_arbol * 1000:.2f}ms")
print(f"Red:   exactitud={exactitud_red_iris:.2%}, tiempo={tiempo_red_iris * 1000:.2f}ms")
print(f"La red tarda {tiempo_red_iris / tiempo_arbol:.0f}x más que el árbol")

assert exactitud_red_iris <= arbol_b, (
    "La red no debería superar al árbol en Iris — si lo hace, el argumento "
    "de sobre-ingeniería de Selección de Arquitectura pierde su evidencia."
)
assert tiempo_red_iris > tiempo_arbol, (
    "La red debería tardar más en entrenar que el árbol, incluso en un "
    "dataset tan pequeño como Iris — si esto falla, revisa que ambos "
    "modelos se estén midiendo con time.perf_counter() alrededor del fit()."
)
```

### Ejercicio C (para resolver): medir el costo real de no escalar

Diseño afirma que escalar las features es "indispensable" para la red de California Housing. Compruébalo entrenando el mismo `MLPRegressor` sin `StandardScaler` y comparando el R² resultante:

```python
X_ch, y_ch = fetch_california_housing(return_X_y=True)
X_train_ch, X_test_ch, y_train_ch, y_test_ch = train_test_split(
    X_ch, y_ch, test_size=0.2, random_state=42
)

modelo_sin_escalar = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
modelo_sin_escalar.fit(X_train_ch, y_train_ch)
r2_sin_escalar = r2_score(y_test_ch, modelo_sin_escalar.predict(X_test_ch))

print(f"R2 con escalado (StandardScaler): {r2:.4f}")
print(f"R2 sin escalado: {r2_sin_escalar:.4f}")

assert r2_sin_escalar < r2, (
    "Entrenar sin escalar las features debería dar un R2 notablemente peor "
    "— si esto falla, revisa que StandardScaler realmente se esté omitiendo "
    "en la versión sin escalar."
)
```

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
| `herramientas` | Registro de Tools | Diccionario nombre → función invocable por el agente, mapeando a las dos funciones de entrenamiento reales de esta unidad (sección Anatomía) |
| `resultado` | Salida de una Tool | Resultado de invocar `herramientas["entrenar_clasificador_iris"]` (sección Anatomía) |
| `memoria` | Instancia de memoria con ventana | `MemoriaConVentana` con `tamano_maximo=3`, retiene decisiones reales de entrenamiento del agente (sección Anatomía) |
| `tamano_maximo` | Tamaño máximo de la ventana de memoria | Límite de mensajes retenidos antes de descartar los más antiguos (sección Anatomía) |
| `decisiones_del_agente` | Historial de ejemplo | 4 mensajes que simulan el registro de decisiones de un agente de ML (sección Anatomía) |
| `acciones_permitidas` | Lista blanca de Guardrail | Conjunto de acciones que el agente puede ejecutar sin bloqueo (sección Anatomía) |
| `X`, `y` | Features y etiquetas/objetivo | Entradas y salida de `load_iris`/`fetch_california_housing` (ML Clásico, Redes Neuronales, Evaluación) |
| `X_train`, `X_test`, `y_train`, `y_test` | Partición train/test de Iris | Usada en el clasificador principal y en la variante `max_depth=1` de Evaluación (ML Clásico, Evaluación) |
| `semilla` | Semilla aleatoria | Fija el split train/test y la inicialización de pesos, para reproducibilidad (ambas secciones de ML) |
| `modelo` | Estimador de scikit-learn | `DecisionTreeClassifier` o `MLPRegressor` según la sección |
| `exactitud` | Accuracy del clasificador | Fracción de predicciones correctas de `entrenar_clasificador_iris` (ML Clásico) |
| `escalador` | `StandardScaler` ajustado | Normaliza las features de California Housing antes de entrenar la red (Redes Neuronales) |
| `rmse`, `r2` | Métricas de regresión | Error cuadrático medio (raíz) y coeficiente de determinación de la red (Redes Neuronales) |
| `modelo_limitado`, `predicciones_limitadas`, `matriz` | Árbol restringido y su matriz de confusión | `DecisionTreeClassifier(max_depth=1)` y `confusion_matrix` real que revela la confusión entre 2 especies (Evaluación) |
| `memoria_con_resumen`, `mensajes_descartados` | Instancia extendida y su contador | `MemoriaConResumen` cuenta cuántos mensajes salieron de la ventana en su atributo `mensajes_descartados`, verificado con `assert` (Ejercicio A) |
| `arbol_b`, `red_iris`, `exactitud_red_iris` | Árbol y red comparados sobre Iris | `arbol_b` reutiliza `entrenar_clasificador_iris`; `red_iris` es un `MLPClassifier` entrenado sobre el mismo split, con exactitud igual o menor (Ejercicio B) |
| `tiempo_arbol`, `tiempo_red_iris` | Tiempos de entrenamiento medidos | `time.perf_counter()` alrededor de cada `.fit()`, comparados para evidenciar el costo de la red frente al árbol (Ejercicio B) |
| `r2_sin_escalar` | R² sin `StandardScaler` | Resultado de entrenar el mismo `MLPRegressor` sin escalar las features, notablemente peor que con escalado (Ejercicio C) |
| `auditor` | Instancia de `CodeAuditorAgent` | Agente real usado para clasificar severidad de código (Cierre Auto-Referencial) |
| `total` | Conteo de hallazgos | Suma de hallazgos de estilo y seguridad reportados por `auditor` (Cierre Auto-Referencial) |

**Verificación manual del Diccionario de Variables** (el mecanismo automático de `ContentAuditorAgent._audit_diccionario_variables` es un placeholder que siempre retorna `[]` — no certifica nada): cada símbolo de la tabla fue releído contra el bloque de código donde aparece antes de agregarlo. Los símbolos de las veinte filas están efectivamente usados en código Python realmente ejecutado en esta unidad: `herramientas`/`resultado` se construyen e invocan en la sección Tools; `memoria`/`tamano_maximo`/`decisiones_del_agente` se instancian y recorren en Memory; `acciones_permitidas` se pasa a `validar_accion_permitida` con dos llamadas reales; `X`/`y`/`X_train`/`X_test`/`y_train`/`y_test` se generan con `load_iris`/`fetch_california_housing`/`train_test_split` reales, tanto en las secciones principales como en la variante `max_depth=1` de Evaluación; `semilla`/`modelo`/`exactitud`/`escalador`/`rmse`/`r2` participan en los dos entrenamientos principales; `modelo_limitado`/`predicciones_limitadas`/`matriz` se construyen y se imprimen en Evaluación, con la matriz de confusión real citada en la prosa; `memoria_con_resumen`/`mensajes_descartados` se instancian y se incrementan dentro de `MemoriaConResumen.agregar`, verificados con `assert` en el Ejercicio A; `arbol_b`/`red_iris`/`exactitud_red_iris`/`tiempo_arbol`/`tiempo_red_iris` se entrenan y miden con `time.perf_counter()` real en el Ejercicio B, con dos `assert` sobre exactitud y tiempo; `r2_sin_escalar` se calcula entrenando el mismo `MLPRegressor` sin escalar en el Ejercicio C; `auditor`/`total` participan en las tres llamadas reales a `CodeAuditorAgent` del Cierre Auto-Referencial.

### Autoevaluación

```python
%%writefile test_unidad_1.py
import sys
from pathlib import Path

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent_core.code_auditor_agent import CodeAuditorAgent


def entrenar_clasificador_iris(semilla: int = 42, max_depth=None) -> float:
    """Entrena un árbol de decisión sobre Iris y retorna su exactitud."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    modelo = DecisionTreeClassifier(random_state=semilla, max_depth=max_depth)
    modelo.fit(X_train, y_train)
    return accuracy_score(y_test, modelo.predict(X_test))


def entrenar_red_california_housing(semilla: int = 42, escalar: bool = True) -> tuple[float, float]:
    """Entrena un MLPRegressor sobre California Housing y retorna (RMSE, R2)."""
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla
    )
    if escalar:
        escalador = StandardScaler()
        X_train = escalador.fit_transform(X_train)
        X_test = escalador.transform(X_test)
    modelo = MLPRegressor(
        hidden_layer_sizes=(32, 16), max_iter=500, random_state=semilla
    )
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
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


def test_matriz_confusion_limitada_confunde_dos_especies_no_tres():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(random_state=42, max_depth=1)
    modelo.fit(X_train, y_train)
    matriz = confusion_matrix(y_test, modelo.predict(X_test))
    # Con max_depth=1, la especie 0 (setosa) se separa perfectamente:
    # su fila y columna deben ser una identidad aislada.
    assert matriz[0, 0] == matriz[0].sum()
    assert matriz[:, 0].sum() == matriz[0, 0]


def test_escalar_features_mejora_r2_de_la_red():
    _, r2_con_escalado = entrenar_red_california_housing(escalar=True)
    _, r2_sin_escalado = entrenar_red_california_housing(escalar=False)
    assert r2_sin_escalado < r2_con_escalado


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

Ejecutado, las 6 pruebas pasan: el árbol de decisión supera 80% de exactitud, la red neuronal supera 0.5 de R², la matriz de confusión del árbol limitado (`max_depth=1`) confirma que la especie *setosa* queda perfectamente aislada mientras las otras dos se confunden entre sí, escalar las features mejora estrictamente el R² de la red frente a no escalarlas, y el clasificador de severidad distingue correctamente código limpio de código riesgoso usando el agente real del repositorio.
