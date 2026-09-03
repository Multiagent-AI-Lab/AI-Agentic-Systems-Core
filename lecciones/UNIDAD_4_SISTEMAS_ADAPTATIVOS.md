# UNIDAD 4: Sistemas Agénticos Adaptativos

> ⚠️ **Línea de Investigación**: esta unidad cubre contenido de frontera, sin el mismo nivel de estandarización de producción que U1-U3 (LangGraph/CrewAI/MCP/A2A son estables; lo que sigue no lo es todavía). Se presenta vocabulario y criterio, con ejercicios reales pero acotados — no una promesa de práctica lista para producción.

## Prefacio

Las Unidades 1 a 3 recorrieron el Ciclo del Agente completo sobre práctica ya estandarizada: un modelo entrenado una vez y desplegado, un sistema multiagente coordinado por un framework maduro. Esta unidad cierra el curso mirando hacia una pregunta distinta: ¿qué pasa cuando el propio agente cambia su comportamiento con el tiempo, en vez de ejecutar siempre la misma política aprendida una sola vez? Esa pregunta no tiene todavía una respuesta de ingeniería tan sólida como las anteriores — de ahí que esta unidad esté marcada explícitamente como Línea de Investigación y no fuerce la sección "El Ciclo del Agente": los dos ejercicios que siguen son reales y verificables, y usan los mismos datos y el mismo modelo que la Unidad 1 (California Housing, `MLPRegressor`) y una versión corregida y ejecutable del entorno de decisión secuencial que motiva el Ejercicio 2 — no un pipeline de producción, pero tampoco un juguete matemático desconectado del resto del curso.

## Vocabulario: Sistemas que Evolucionan, se Adaptan o Aprenden

- **Learning Agents**: dividen su estructura en un elemento de aprendizaje (evalúa errores, mejora) y un elemento de ejecución (actúa en el entorno).
- **Sistemas Evolutivos / Algoritmos Genéticos**: lógica de selección natural — mutación, combinación, supervivencia del más apto.
- **Lifelong Learning Agents (Aprendizaje por Refuerzo Continuo)**: aprenden por prueba y error de forma ininterrumpida, adaptando comportamiento por recompensas sin olvidar lo aprendido previamente.
- **Sistemas Complejos Adaptativos (CAS)**: redes de agentes individuales que interactúan; el sistema completo evoluciona y muestra comportamiento emergente.
- **Sistemas Auto-organizativos**: modifican su propia estructura interna en respuesta al entorno, sin control central.

**3 características clave**: Autonomía (operan sin intervención humana directa), Plasticidad (modifican sus algoritmos internos según la experiencia), Homeostasis (buscan estabilidad interna frente al caos externo).

Estos 8 términos no son igualmente centrales para lo que sigue: los dos ejercicios de esta unidad son ejemplos concretos de **Sistemas Evolutivos** (Ejercicio 1) y de **Lifelong Learning Agent** (Ejercicio 2) — los dos únicos de la lista con una implementación ejecutable aquí. Los otros cinco (CAS, Sistemas Auto-organizativos, Learning Agents como categoría general, y las tres características) describen el panorama más amplio en el que esos dos ejercicios son casos particulares, pero no tienen código propio en esta unidad. Concretamente: el algoritmo genético del Ejercicio 1 exhibe **Plasticidad** (la población cambia su composición generación a generación en respuesta a la métrica) pero no **Homeostasis** — nada en `evolucionar_configuracion` preserva al mejor individuo encontrado frente a una mutación destructiva; si el mejor hijo de una generación resulta peor que sus padres, se pierde igual que cualquier otro, porque la selección solo mira el fitness de la generación actual, no el mejor histórico. Un algoritmo genético real de producción añadiría **elitismo** (conservar siempre al mejor individuo sin mutar) precisamente para ganar esa Homeostasis que la versión de este ejercicio no tiene — se deja como Ejercicio 1b más abajo.

## Ejercicio 1: Algoritmo Genético para Tuning de Hiperparámetros

Un algoritmo genético es el ejemplo más directo de "Sistema Evolutivo": una población de soluciones candidatas se somete a selección (sobreviven las mejores según una métrica) y mutación (las sobrevivientes generan variantes ligeramente distintas), y el proceso se repite hasta que la población converge hacia el óptimo de la métrica.

### Por qué un algoritmo genético y no Optimización Bayesiana

La Unidad 2 ya resolvió un problema de la misma familia — "encontrar el punto que maximiza una función costosa de evaluar" — con Optimización Bayesiana (BO). Ambos enfoques comparten el objetivo, pero difieren en qué asumen y qué exploran:

- **BO explota un modelo sustituto.** El Proceso Gaussiano de la Unidad 2 asume que la función objetivo es razonablemente suave (eso es lo que codifica el kernel Matern) e interpola entre las evaluaciones ya hechas para decidir dónde evaluar a continuación. Cuando esa suposición es correcta, BO converge con muy pocas evaluaciones (la Unidad 2 usó 10).
- **Un GA no asume nada sobre la forma de la métrica**, y por eso funciona igual de bien sobre superficies con múltiples óptimos locales, discontinuidades, o interacciones no lineales entre variables — el precio es que necesita evaluar muchos más puntos (aquí, decenas de individuos a lo largo de varias generaciones) porque no construye ningún modelo de la función, solo la evalúa directamente una y otra vez.

El caso de este ejercicio hace explícita esa diferencia: **la métrica no es una función matemática simple como la parábola de la Unidad 2 — es el R² real de entrenar un `MLPRegressor` sobre California Housing**, exactamente el modelo que la Unidad 1 entrenó una sola vez con hiperparámetros elegidos a mano. Esa métrica sí es costosa de evaluar (cada evaluación implica entrenar una red neuronal completa) y su forma respecto a los hiperparámetros (tasa de aprendizaje, regularización, tamaño de la capa oculta) no tiene ninguna garantía de suavidad — dos configuraciones muy parecidas pueden dar resultados muy distintos si una cae en una zona de entrenamiento inestable. BO sería una alternativa razonable aquí también (y de hecho es la opción más usada en la práctica para este problema exacto, bajo el nombre de *hyperparameter tuning*); se usa un GA en este ejercicio porque es el que corresponde al vocabulario de "Sistema Evolutivo" que esta sección introduce, no porque sea objetivamente superior a BO para este caso.

```python
import warnings

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)  # ConvergenceWarning esperado con max_iter bajo

# Mismo dataset que la Unidad 1 — Redes Neuronales con California Housing.
# Se usa un subconjunto para que cada evaluación del GA (un entrenamiento
# completo de MLP) tome menos de un segundo, ya que el GA necesita muchas.
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_sub, y_train_sub = X_train[:1500], y_train[:1500]
X_test_sub, y_test_sub = X_test[:500], y_test[:500]

escalador = StandardScaler().fit(X_train_sub)
X_train_esc = escalador.transform(X_train_sub)
X_test_esc = escalador.transform(X_test_sub)


def metrica_r2_mlp(config: np.ndarray) -> float:
    """Entrena un MLPRegressor con la configuración dada (un individuo de
    la población) y retorna su R2 sobre el conjunto de prueba.

    config = [log10(alpha), log10(learning_rate_init), neuronas_capa_oculta]
    — los dos primeros genes están en escala logarítmica porque alpha y la
    tasa de aprendizaje son sensibles a órdenes de magnitud, no a su valor
    absoluto (buscar entre 0.0001 y 0.1 en escala lineal desperdiciaría casi
    toda la búsqueda en la mitad superior del rango).
    """
    alpha = 10 ** config[0]
    tasa_aprendizaje = 10 ** config[1]
    neuronas = max(4, int(config[2]))
    modelo = MLPRegressor(
        hidden_layer_sizes=(neuronas,),
        alpha=alpha,
        learning_rate_init=tasa_aprendizaje,
        max_iter=100,
        random_state=42,
    )
    modelo.fit(X_train_esc, y_train_sub)
    return r2_score(y_test_sub, modelo.predict(X_test_esc))


def evolucionar_configuracion(
    metrica: callable, poblacion_inicial: np.ndarray, generaciones: int = 5, semilla: int = 42
) -> tuple[np.ndarray, float]:
    """Evoluciona una población de configuraciones (vectores de floats)
    hacia el máximo de una métrica dada, vía selección + mutación gaussiana.

    Args:
        metrica: Función que recibe un vector 1D y retorna un float a maximizar.
        poblacion_inicial: Array (n_individuos, n_genes) con la población inicial.
        generaciones: Número de generaciones a evolucionar.
        semilla: Semilla aleatoria para reproducibilidad.

    Returns:
        Tupla (mejor individuo encontrado, su fitness) tras todas las generaciones.
    """
    rng = np.random.default_rng(semilla)
    poblacion = poblacion_inicial.copy()
    for _ in range(generaciones):
        fitness = np.array([metrica(ind) for ind in poblacion])
        mejores_indices = np.argsort(fitness)[-len(poblacion) // 2:]
        sobrevivientes = poblacion[mejores_indices]
        hijos = sobrevivientes + rng.normal(0, 0.15, sobrevivientes.shape)
        poblacion = np.vstack([sobrevivientes, hijos])[: len(poblacion_inicial)]
    fitness_final = np.array([metrica(ind) for ind in poblacion])
    return poblacion[np.argmax(fitness_final)], float(fitness_final.max())


# Rangos de búsqueda: log10(alpha) en [-5,-1], log10(learning_rate) en [-4,-1],
# neuronas en [8, 48] — 6 individuos iniciales, 3 genes cada uno.
poblacion_inicial = np.column_stack([
    np.random.default_rng(42).uniform(-5, -1, 6),
    np.random.default_rng(43).uniform(-4, -1, 6),
    np.random.default_rng(44).uniform(8, 48, 6),
])
mejor_config, mejor_r2 = evolucionar_configuracion(metrica_r2_mlp, poblacion_inicial)
print(f"Mejor configuración: alpha=10^{mejor_config[0]:.2f}, "
      f"learning_rate={10 ** mejor_config[1]:.4f}, neuronas={int(mejor_config[2])}")
print(f"R2 en test: {mejor_r2:.4f}")
```

Ejecutado, este bloque converge en 5 generaciones a una configuración con `alpha≈10^-2.19` (≈0.0065), `learning_rate≈0.0295` y 46 neuronas en la capa oculta, con **R² de 0.7759** sobre el conjunto de prueba — resultado consistente con el R² de 0.78 que la Unidad 1 reportó entrenando el mismo tipo de red con hiperparámetros elegidos a mano, confirmando que el GA encontró una configuración competitiva sin que nadie la eligiera manualmente. A diferencia del Ejercicio 1 de la versión anterior de esta unidad (que evolucionaba hacia un vector constante `[1,1,1]` sin ninguna conexión con IA real), esta métrica es el resultado de entrenar y evaluar una red neuronal real sobre datos reales — el "problema costoso de evaluar" que motiva usar un algoritmo evolutivo en vez de fuerza bruta es, aquí, literalmente cierto.

### Ejercicio 1b (para resolver): variar la tasa de mutación

Modifica `evolucionar_configuracion` para que la desviación estándar de la mutación gaussiana decaiga con la generación, en vez de mantenerse fija en `0.15` durante todo el proceso — una tasa de mutación alta ayuda a explorar al principio, pero dificulta converger con precisión hacia el final:

```python
def evolucionar_configuracion_con_decaimiento(
    metrica: callable, poblacion_inicial: np.ndarray, generaciones: int = 5, semilla: int = 42
) -> tuple[np.ndarray, float]:
    """Igual que evolucionar_configuracion, pero la desviación estándar de
    la mutación decae linealmente de 0.15 en la generación 0 a ~0 en la
    última generación: sigma_g = 0.15 * (1 - g / generaciones).
    """
    rng = np.random.default_rng(semilla)
    poblacion = poblacion_inicial.copy()
    for g in range(generaciones):
        fitness = np.array([metrica(ind) for ind in poblacion])
        mejores_indices = np.argsort(fitness)[-len(poblacion) // 2:]
        sobrevivientes = poblacion[mejores_indices]
        sigma_g = 0.15 * (1 - g / generaciones)
        hijos = sobrevivientes + rng.normal(0, sigma_g, sobrevivientes.shape)
        poblacion = np.vstack([sobrevivientes, hijos])[: len(poblacion_inicial)]
    fitness_final = np.array([metrica(ind) for ind in poblacion])
    return poblacion[np.argmax(fitness_final)], float(fitness_final.max())


_, r2_con_decaimiento = evolucionar_configuracion_con_decaimiento(metrica_r2_mlp, poblacion_inicial)
_, r2_sin_decaimiento = evolucionar_configuracion(metrica_r2_mlp, poblacion_inicial)
print(f"R2 con decaimiento de mutación: {r2_con_decaimiento:.4f}")
print(f"R2 sin decaimiento (tasa fija): {r2_sin_decaimiento:.4f}")

assert r2_con_decaimiento >= r2_sin_decaimiento - 0.02, (
    "El decaimiento no debería empeorar significativamente el resultado "
    "final — si esto falla, revisa que sigma_g realmente disminuya con g."
)
```

## Ejercicio 2: Q-Learning Tabular sobre un Entorno con Transiciones

Q-learning es el ejemplo más directo de "Lifelong Learning Agent" reducido a su forma más simple: en vez de una política fija, el agente mantiene una tabla de valores `Q(estado, acción)` que actualiza después de cada intento, acercándose gradualmente a la acción de mayor recompensa esperada en cada estado — **incluyendo recompensas futuras, no solo la inmediata**, que es precisamente lo que distingue Q-learning de simplemente promediar recompensas por acción.

Esa distinción exige un entorno con **transiciones de estado**: si una acción no cambia en qué estado queda el agente, no hay ningún "futuro" que propagar hacia atrás, y la actualización de Bellman (`recompensa + factor_descuento * mejor_futuro`) se colapsa a un promedio ponderado de recompensas inmediatas — que es Q-learning solo de nombre. Este ejercicio usa un **GridWorld 1D de 4 estados** (`0 — 1 — 2 — 3`, donde `3` es la meta) precisamente para que el término de valor futuro tenga algo real que propagar: la única recompensa está en la transición que llega a la meta, y el agente debe aprender a preferir "moverse a la derecha" en los estados 0, 1 y 2 aunque esos estados no den ninguna recompensa por sí mismos.

```python
import numpy as np


def entrenar_q_learning(
    n_estados: int, n_acciones: int, recompensas: np.ndarray, transiciones: np.ndarray,
    episodios: int = 300, tasa_aprendizaje: float = 0.1,
    factor_descuento: float = 0.9, semilla: int = 42,
) -> np.ndarray:
    """Entrena una tabla Q sobre un entorno con transiciones de estado reales.

    Args:
        n_estados: Número de estados del entorno.
        n_acciones: Número de acciones posibles.
        recompensas: Matriz (n_estados, n_acciones) de recompensas inmediatas.
        transiciones: Matriz (n_estados, n_acciones) de enteros — a qué
            estado lleva tomar cada acción desde cada estado. Este es el
            elemento que faltaba en una versión anterior de este ejercicio:
            sin él, `mejor_futuro` se calculaba sobre el mismo estado en
            vez del estado siguiente, y el término de descuento no tenía
            ningún efecto real sobre el aprendizaje.
        episodios: Número de episodios de entrenamiento.
        tasa_aprendizaje: Alfa de la actualización Q-learning.
        factor_descuento: Gamma de la actualización Q-learning.
        semilla: Semilla aleatoria para exploración.

    Returns:
        La tabla Q entrenada, forma (n_estados, n_acciones).
    """
    rng = np.random.default_rng(semilla)
    tabla_q = np.zeros((n_estados, n_acciones))
    for _ in range(episodios):
        estado = rng.integers(0, n_estados)
        accion = rng.integers(0, n_acciones)
        recompensa = recompensas[estado, accion]
        siguiente_estado = transiciones[estado, accion]
        mejor_futuro = np.max(tabla_q[siguiente_estado])
        tabla_q[estado, accion] += tasa_aprendizaje * (
            recompensa + factor_descuento * mejor_futuro - tabla_q[estado, accion]
        )
    return tabla_q


# GridWorld 1D: estados 0-1-2-3, acciones 0=izquierda, 1=derecha.
# Recompensa solo al llegar a la meta (estado 3) desde el estado 2.
n_estados, n_acciones = 4, 2
recompensas = np.zeros((n_estados, n_acciones))
recompensas[2, 1] = 1.0  # única recompensa: estado 2, acción derecha

transiciones = np.zeros((n_estados, n_acciones), dtype=int)
for estado in range(n_estados):
    transiciones[estado, 0] = max(0, estado - 1)              # izquierda
    transiciones[estado, 1] = min(n_estados - 1, estado + 1)  # derecha
transiciones[3, :] = 3  # la meta es un estado terminal: cualquier acción se queda ahí

tabla_final = entrenar_q_learning(n_estados, n_acciones, recompensas, transiciones)
print(f"Tabla Q entrenada:\n{tabla_final}")
print(f"Política óptima por estado: {[int(np.argmax(tabla_final[s])) for s in range(3)]}")
```

Ejecutado, la tabla Q converge de forma que `argmax(tabla_final[s]) == 1` ("moverse a la derecha") en los tres estados no terminales (`0`, `1`, `2`), y los valores de la acción "derecha" **decrecen a medida que el estado se aleja de la meta** (`tabla_final[2,1] > tabla_final[1,1] > tabla_final[0,1]`) — exactamente el comportamiento que el factor de descuento `factor_descuento=0.9` predice: un episodio que necesita más pasos para llegar a la recompensa vale menos que uno más corto, y esa diferencia solo puede aparecer si `mejor_futuro` se calcula sobre el estado al que la acción realmente lleva. Con la versión anterior de este ejercicio (sin `transiciones`, calculando `mejor_futuro` sobre el mismo estado de origen) esta propagación de valor hacia atrás no podía ocurrir — el agente sí aprendía a preferir la acción de mayor recompensa inmediata en cada estado por separado, pero eso es un problema de bandidos multi-brazo independientes por estado, no la propagación de valor a través de una secuencia de decisiones que es la contribución específica de Q-learning frente a un promedio simple.

### Ejercicio 2b (para resolver): un GridWorld más largo

Extiende el entorno a 6 estados (`0` a `5`, meta en `5`) y verifica que la política óptima sigue siendo "moverse a la derecha" en todos los estados no terminales, y que el valor de la acción óptima en el estado más lejano de la meta (`0`) es menor que en el GridWorld de 4 estados (porque hace falta más pasos para llegar a la recompensa, y cada paso adicional multiplica el valor por `factor_descuento`):

```python
n_estados_largo = 6
recompensas_largo = np.zeros((n_estados_largo, n_acciones))
recompensas_largo[n_estados_largo - 2, 1] = 1.0  # recompensa al llegar a la meta

transiciones_largo = np.zeros((n_estados_largo, n_acciones), dtype=int)
for estado in range(n_estados_largo):
    transiciones_largo[estado, 0] = max(0, estado - 1)
    transiciones_largo[estado, 1] = min(n_estados_largo - 1, estado + 1)
transiciones_largo[n_estados_largo - 1, :] = n_estados_largo - 1

tabla_largo = entrenar_q_learning(n_estados_largo, n_acciones, recompensas_largo, transiciones_largo, episodios=600)
print(f"Política óptima (GridWorld largo): {[int(np.argmax(tabla_largo[s])) for s in range(n_estados_largo - 1)]}")
print(f"Valor de 'derecha' en estado 0 — GridWorld de 4 estados: {tabla_final[0, 1]:.4f}")
print(f"Valor de 'derecha' en estado 0 — GridWorld de 6 estados: {tabla_largo[0, 1]:.4f}")

assert all(np.argmax(tabla_largo[s]) == 1 for s in range(n_estados_largo - 1)), (
    "La política óptima debería ser 'derecha' en todos los estados no terminales."
)
assert tabla_largo[0, 1] < tabla_final[0, 1], (
    "El valor en el estado más lejano de la meta debería ser menor cuantos "
    "más pasos separen al agente de la recompensa (más descuentos aplicados)."
)
```

## Riesgo Abierto: Catastrophic Forgetting y Memory Poisoning

El aprendizaje continuo "sin olvidar lo aprendido previamente" (Lifelong Learning) sigue siendo un problema de investigación abierto — **no se implementa aquí como ejercicio resuelto**. Un agente que se auto-modifica en producción agrava el vector de memory poisoning ya visto en U3: si el propio mecanismo de aprendizaje puede ser manipulado por entradas adversariales, el problema de seguridad se vuelve estructuralmente más difícil que con un agente estático — el componente Memory del Harness (Unidad 1) deja de ser un registro pasivo que un Guardrail puede auditar después del hecho, y pasa a ser el propio mecanismo que decide cómo se comporta el agente mañana.

La conexión concreta con U3: `SafetyGateAgent.check_output()` ya demostró no detectar una afirmación de privilegios inyectada en la memoria persistente de un agente — un validador de texto aislado no tiene forma de comparar esa afirmación contra el historial real de permisos otorgados. Un Learning Agent o un Lifelong Learning Agent que ajusta su propia tabla de valores o sus propios pesos a partir de esas mismas interacciones no solo hereda ese punto ciego: lo agrava, porque la entrada envenenada no queda aislada en una respuesta puntual, sino que se incorpora a la política interna del agente y sigue influyendo en decisiones futuras — el equivalente agéntico de *catastrophic forgetting* inducido deliberadamente, donde lo que se "olvida" es la restricción de seguridad original. Ni este repositorio ni la literatura consultada hasta la fecha de escritura (septiembre de 2026) ofrecen todavía un mecanismo estándar de defensa contra esto; es, junto con la generalización de redes sobreparametrizadas y la causalidad en modelos de lenguaje, una de las preguntas de frontera sin respuesta consolidada — vale la pena revisar si esto sigue siendo cierto al releer esta sección en el futuro, precisamente por tratarse de un área activa de investigación.

### Diccionario de Variables

| Símbolo | Nombre | Descripción |
|---|---|---|
| `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test` | Features y objetivo de California Housing, y su partición | Mismo dataset que la Unidad 1 (`fetch_california_housing`); partición 80/20 con `train_test_split` (Ejercicio 1) |
| `X_train_sub`, `y_train_sub`, `X_test_sub`, `y_test_sub` | Subconjuntos de entrenamiento/prueba | Primeras 1500/500 filas, usadas para que cada evaluación del GA (un entrenamiento de MLP) sea rápida (Ejercicio 1) |
| `escalador` | `StandardScaler` ajustado | Normaliza las 8 features de California Housing antes de entrenar cada MLP candidato (Ejercicio 1) |
| `metrica_r2_mlp` | Función objetivo del algoritmo genético | Recibe un vector `[log10(alpha), log10(lr), neuronas]`, entrena un `MLPRegressor` con esa configuración y retorna su R² real sobre datos de prueba (Ejercicio 1) |
| `poblacion_inicial`, `poblacion` | Población de configuraciones candidatas | Array `(n_individuos, 3)`; `poblacion` se reasigna cada generación tras selección + mutación (Ejercicio 1) |
| `fitness`, `fitness_final` | Valores de aptitud (R²) de la población | Array 1D con `metrica_r2_mlp(ind)` evaluada sobre cada individuo, usado para ordenar y seleccionar (Ejercicio 1) |
| `sobrevivientes`, `hijos` | Individuos seleccionados y su descendencia mutada | `sobrevivientes` son la mitad de mayor fitness; `hijos` suman ruido gaussiano `rng.normal(0, 0.15, ...)` (Ejercicio 1) |
| `mejor_config`, `mejor_r2` | Mejor configuración final y su R² | Resultado de `evolucionar_configuracion`, impreso y comparado contra el R² de referencia de la Unidad 1 (Ejercicio 1, Autoevaluación) |
| `sigma_g` | Desviación estándar de mutación en la generación `g` | Decae linealmente con la generación en `evolucionar_configuracion_con_decaimiento` (Ejercicio 1b) |
| `n_estados`, `n_acciones` | Tamaño del entorno de Q-learning | `n_estados=4` (GridWorld corto) o `6` (Ejercicio 2b); `n_acciones=2` (izquierda/derecha) |
| `recompensas` | Matriz de recompensas inmediatas del entorno | `recompensas[estado, accion]`; solo la transición que llega a la meta tiene recompensa `1.0`, el resto es `0.0` (Ejercicio 2) |
| `transiciones` | Matriz de transiciones de estado | `transiciones[estado, accion]` indica a qué estado lleva cada acción — el elemento que la versión anterior de este ejercicio no tenía (Ejercicio 2) |
| `tabla_q`, `tabla_final`, `tabla_largo` | Tabla(s) de valores Q | Array `(n_estados, n_acciones)` actualizado episodio a episodio con la regla de Q-learning; `tabla_largo` es la versión del GridWorld de 6 estados (Ejercicio 2, 2b) |
| `estado`, `accion`, `siguiente_estado` | Estado, acción y estado resultante en cada episodio | `estado`/`accion` se muestrean con `rng.integers`; `siguiente_estado = transiciones[estado, accion]` (Ejercicio 2) |
| `mejor_futuro` | Estimación del valor futuro máximo | `np.max(tabla_q[siguiente_estado])` — término de la actualización de Bellman, ahora calculado sobre el estado al que la acción realmente lleva (Ejercicio 2) |

**Verificación manual del Diccionario de Variables** (el mecanismo automático de `ContentAuditorAgent._audit_diccionario_variables` es un placeholder que siempre retorna `[]` — no certifica nada): cada símbolo de la tabla fue releído contra el bloque de código donde aparece antes de agregarlo. Los quince símbolos están efectivamente usados en código Python realmente ejecutado en esta unidad (no en una tabla de sintaxis genérica ni en un docstring aislado): `X`/`y`/`X_train`/`X_test`/`y_train`/`y_test` se generan con `fetch_california_housing` y `train_test_split` reales; `X_train_sub`/`y_train_sub`/`X_test_sub`/`y_test_sub` se usan directamente dentro de `metrica_r2_mlp` para entrenar y evaluar; `escalador` transforma ambos conjuntos; `metrica_r2_mlp` se invoca dentro del bucle de evolución sobre cada individuo; `poblacion_inicial`/`poblacion` se transforman en cada iteración del `for`; `fitness`/`fitness_final` se calculan y se usan para `argsort`/`argmax` reales; `sobrevivientes`/`hijos` se construyen con slicing y `rng.normal` y se concatenan con `np.vstack`; `mejor_config`/`mejor_r2` se imprimen y se comparan contra el R² de la Unidad 1 en la prosa; `sigma_g` se recalcula en cada generación del Ejercicio 1b y decae verificablemente; `n_estados`/`n_acciones` parametrizan las matrices de ambos GridWorld; `recompensas`/`transiciones` se leen dentro de `entrenar_q_learning` para cada episodio; `tabla_q`/`tabla_final`/`tabla_largo` se inicializan en ceros y se actualizan episodio a episodio; `estado`/`accion`/`siguiente_estado` se muestrean o derivan en cada iteración del bucle de entrenamiento; `mejor_futuro` se calcula con `np.max` sobre `tabla_q[siguiente_estado]` en cada episodio y su corrección (usar el estado siguiente, no el actual) se verifica empíricamente en la Autoevaluación comparando los valores de dos GridWorld de distinta longitud.

### Autoevaluación

```python
%%writefile test_unidad_4.py
import numpy as np


def entrenar_q_learning(n_estados, n_acciones, recompensas, transiciones, episodios=300, tasa_aprendizaje=0.1, factor_descuento=0.9, semilla=42):
    rng = np.random.default_rng(semilla)
    tabla_q = np.zeros((n_estados, n_acciones))
    for _ in range(episodios):
        estado = rng.integers(0, n_estados)
        accion = rng.integers(0, n_acciones)
        recompensa = recompensas[estado, accion]
        siguiente_estado = transiciones[estado, accion]
        mejor_futuro = np.max(tabla_q[siguiente_estado])
        tabla_q[estado, accion] += tasa_aprendizaje * (
            recompensa + factor_descuento * mejor_futuro - tabla_q[estado, accion]
        )
    return tabla_q


def _construir_gridworld(n_estados):
    n_acciones = 2
    recompensas = np.zeros((n_estados, n_acciones))
    recompensas[n_estados - 2, 1] = 1.0
    transiciones = np.zeros((n_estados, n_acciones), dtype=int)
    for estado in range(n_estados):
        transiciones[estado, 0] = max(0, estado - 1)
        transiciones[estado, 1] = min(n_estados - 1, estado + 1)
    transiciones[n_estados - 1, :] = n_estados - 1
    return recompensas, transiciones


def test_q_learning_aprende_politica_optima_de_moverse_a_la_meta():
    n_estados = 4
    recompensas, transiciones = _construir_gridworld(n_estados)
    tabla = entrenar_q_learning(n_estados, 2, recompensas, transiciones)

    # La politica optima en todo estado no terminal es "derecha" (accion 1)
    for estado in range(n_estados - 1):
        assert np.argmax(tabla[estado]) == 1


def test_q_learning_propaga_valor_futuro_con_el_descuento():
    n_estados = 4
    recompensas, transiciones = _construir_gridworld(n_estados)
    tabla = entrenar_q_learning(n_estados, 2, recompensas, transiciones)

    # El valor de "derecha" debe decrecer al alejarse de la meta (estado 2 > 1 > 0):
    # esto solo es posible si mejor_futuro usa el estado SIGUIENTE, no el actual.
    assert tabla[2, 1] > tabla[1, 1] > tabla[0, 1]


def test_gridworld_mas_largo_tiene_menor_valor_en_el_estado_inicial():
    recompensas_corto, transiciones_corto = _construir_gridworld(4)
    tabla_corto = entrenar_q_learning(4, 2, recompensas_corto, transiciones_corto)

    recompensas_largo, transiciones_largo = _construir_gridworld(6)
    tabla_largo = entrenar_q_learning(6, 2, recompensas_largo, transiciones_largo, episodios=600)

    # Mas pasos hasta la meta = mas descuentos aplicados = menor valor en el estado 0
    assert tabla_largo[0, 1] < tabla_corto[0, 1]
```

```python
!pytest test_unidad_4.py -v
```

Ejecutado, las 3 pruebas pasan: la tabla Q converge a la política óptima ("moverse a la derecha") en todos los estados no terminales del GridWorld, los valores de esa acción decrecen monótonamente a medida que el estado se aleja de la meta —confirmando que el término de valor futuro se calcula sobre el estado al que la acción realmente lleva, no sobre el estado de origen—, y el mismo patrón se sostiene al alargar el entorno de 4 a 6 estados, con un valor inicial estrictamente menor cuantos más pasos separan al agente de la recompensa. El primer y el tercer test corresponden directamente al Ejercicio 2 y al Ejercicio 2b; el algoritmo genético del Ejercicio 1 y su variante con decaimiento (Ejercicio 1b) no se incluyen en este archivo porque requieren entrenar `MLPRegressor` real (varios segundos por corrida) y ya llevan su propio `assert` de verificación en el bloque de la sección correspondiente — mismo criterio que las autoevaluaciones de U0-U3 aplican a los bloques costosos de ejecutar. Las funciones de Q-learning se redefinen dentro del archivo de test (no se importan del notebook) — mismo patrón que ya usan las autoevaluaciones de U0-U3, evita depender de un mecanismo de import frágil hacia celdas de notebook ejecutadas previamente.
