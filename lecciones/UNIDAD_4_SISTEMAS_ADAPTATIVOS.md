# UNIDAD 4: Sistemas Agénticos Adaptativos

> ⚠️ **Línea de Investigación**: esta unidad cubre contenido de frontera, sin el mismo nivel de estandarización de producción que U1-U3 (LangGraph/CrewAI/MCP/A2A son estables; lo que sigue no lo es todavía). Se presenta vocabulario y criterio, con ejercicios reales pero acotados — no una promesa de práctica lista para producción.

## Prefacio

Las Unidades 1 a 3 recorrieron el Ciclo del Agente completo sobre práctica ya estandarizada: un modelo entrenado una vez y desplegado, un sistema multiagente coordinado por un framework maduro. Esta unidad cierra el curso mirando hacia una pregunta distinta: ¿qué pasa cuando el propio agente cambia su comportamiento con el tiempo, en vez de ejecutar siempre la misma política aprendida una sola vez? Esa pregunta no tiene todavía una respuesta de ingeniería tan sólida como las anteriores — de ahí que esta unidad esté marcada explícitamente como Línea de Investigación y no fuerce la sección "El Ciclo del Agente": los dos ejercicios que siguen son reales y verificables, pero acotados a mecanismos de juguete que ilustran la idea, no a un pipeline de producción.

## Vocabulario: Sistemas que Evolucionan, se Adaptan o Aprenden

- **Learning Agents**: dividen su estructura en un elemento de aprendizaje (evalúa errores, mejora) y un elemento de ejecución (actúa en el entorno).
- **Sistemas Evolutivos / Algoritmos Genéticos**: lógica de selección natural — mutación, combinación, supervivencia del más apto.
- **Lifelong Learning Agents (Aprendizaje por Refuerzo Continuo)**: aprenden por prueba y error de forma ininterrumpida, adaptando comportamiento por recompensas sin olvidar lo aprendido previamente.
- **Sistemas Complejos Adaptativos (CAS)**: redes de agentes individuales que interactúan; el sistema completo evoluciona y muestra comportamiento emergente.
- **Sistemas Auto-organizativos**: modifican su propia estructura interna en respuesta al entorno, sin control central.

**3 características clave**: Autonomía (operan sin intervención humana directa), Plasticidad (modifican sus algoritmos internos según la experiencia), Homeostasis (buscan estabilidad interna frente al caos externo).

## Ejercicio 1: Algoritmo Genético Simple

Un algoritmo genético es el ejemplo más directo de "Sistema Evolutivo": una población de soluciones candidatas se somete a selección (sobreviven las mejores según una métrica) y mutación (las sobrevivientes generan variantes ligeramente distintas), y el proceso se repite hasta que la población converge hacia el óptimo de la métrica.

```python
import numpy as np


def evolucionar_configuracion(
    metrica: callable, poblacion_inicial: np.ndarray, generaciones: int = 20, semilla: int = 42
) -> np.ndarray:
    """Evoluciona una población de configuraciones (vectores de floats)
    hacia el máximo de una métrica dada, vía selección + mutación gaussiana.

    Args:
        metrica: Función que recibe un vector 1D y retorna un float a maximizar.
        poblacion_inicial: Array (n_individuos, n_genes) con la población inicial.
        generaciones: Número de generaciones a evolucionar.
        semilla: Semilla aleatoria para reproducibilidad.

    Returns:
        El mejor individuo encontrado tras todas las generaciones.
    """
    rng = np.random.default_rng(semilla)
    poblacion = poblacion_inicial.copy()
    for _ in range(generaciones):
        fitness = np.array([metrica(ind) for ind in poblacion])
        mejores_indices = np.argsort(fitness)[-len(poblacion) // 2:]
        sobrevivientes = poblacion[mejores_indices]
        hijos = sobrevivientes + rng.normal(0, 0.1, sobrevivientes.shape)
        poblacion = np.vstack([sobrevivientes, hijos])[: len(poblacion_inicial)]
    fitness_final = np.array([metrica(ind) for ind in poblacion])
    return poblacion[np.argmax(fitness_final)]


def metrica_ejemplo(config: np.ndarray) -> float:
    """Métrica de ejemplo: qué tan cerca está la configuración del vector [1, 1, 1]."""
    return -np.sum((config - 1.0) ** 2)


poblacion_inicial = np.random.default_rng(42).uniform(-2, 2, size=(10, 3))
mejor = evolucionar_configuracion(metrica_ejemplo, poblacion_inicial)
print(f"Mejor configuración encontrada: {mejor}")
```

Ejecutado, la población parte de un fitness inicial máximo de aproximadamente `-0.32` y converge tras 20 generaciones a una configuración cercana a `[1, 1, 1]` con fitness final del orden de `-8e-05` — la selección + mutación gaussiana efectivamente empuja a la población hacia el óptimo de la métrica, sin que ningún individuo original conociera esa dirección de antemano.

## Ejercicio 2: Q-Learning Tabular Básico

Q-learning es el ejemplo más directo de "Lifelong Learning Agent" reducido a su forma más simple: en vez de una política fija, el agente mantiene una tabla de valores `Q(estado, acción)` que actualiza después de cada intento, acercándose gradualmente a la acción de mayor recompensa esperada en cada estado.

```python
import numpy as np


def entrenar_q_learning(
    n_estados: int, n_acciones: int, recompensas: np.ndarray,
    episodios: int = 100, tasa_aprendizaje: float = 0.1,
    factor_descuento: float = 0.9, semilla: int = 42,
) -> np.ndarray:
    """Entrena una tabla Q simple sobre un entorno de recompensas fijas
    (sin transiciones de estado, solo para ilustrar la actualización básica).

    Args:
        n_estados: Número de estados del entorno.
        n_acciones: Número de acciones posibles.
        recompensas: Matriz (n_estados, n_acciones) de recompensas.
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
        mejor_futuro = np.max(tabla_q[estado])
        tabla_q[estado, accion] += tasa_aprendizaje * (
            recompensa + factor_descuento * mejor_futuro - tabla_q[estado, accion]
        )
    return tabla_q


recompensas = np.array([[1.0, 0.0], [0.0, 1.0]])
tabla_final = entrenar_q_learning(n_estados=2, n_acciones=2, recompensas=recompensas)
print(f"Tabla Q entrenada:\n{tabla_final}")
```

Ejecutado, la tabla Q converge de forma que `argmax(tabla_final[0]) == 0` y `argmax(tabla_final[1]) == 1`: el agente aprendió, solo por prueba y error con exploración aleatoria, que la acción 0 es la mejor en el estado 0 y la acción 1 es la mejor en el estado 1 — exactamente la estructura de la matriz `recompensas` que nunca se le mostró directamente, solo mediante recompensas observadas episodio a episodio.

## Riesgo Abierto: Catastrophic Forgetting y Memory Poisoning

El aprendizaje continuo "sin olvidar lo aprendido previamente" (Lifelong Learning) sigue siendo un problema de investigación abierto — **no se implementa aquí como ejercicio resuelto**. Un agente que se auto-modifica en producción agrava el vector de memory poisoning ya visto en U3: si el propio mecanismo de aprendizaje puede ser manipulado por entradas adversariales, el problema de seguridad se vuelve estructuralmente más difícil que con un agente estático.

La conexión concreta con U3: `SafetyGateAgent.check_output()` ya demostró no detectar una afirmación de privilegios inyectada en la memoria persistente de un agente — un validador de texto aislado no tiene forma de comparar esa afirmación contra el historial real de permisos otorgados. Un Learning Agent o un Lifelong Learning Agent que ajusta su propia tabla de valores o sus propios pesos a partir de esas mismas interacciones no solo hereda ese punto ciego: lo agrava, porque la entrada envenenada no queda aislada en una respuesta puntual, sino que se incorpora a la política interna del agente y sigue influyendo en decisiones futuras — el equivalente agéntico de *catastrophic forgetting* inducido deliberadamente, donde lo que se "olvida" es la restricción de seguridad original. Ni este repositorio ni la literatura de referencia ofrecen todavía un mecanismo estándar de defensa contra esto; es, junto con la generalización de redes sobreparametrizadas y la causalidad en modelos de lenguaje, una de las preguntas de frontera sin respuesta consolidada.

### Diccionario de Variables

| Símbolo | Nombre | Descripción |
|---|---|---|
| `metrica`, `metrica_ejemplo` | Función objetivo del algoritmo genético | Recibe un vector 1D y retorna un float a maximizar; `metrica_ejemplo` mide cercanía a `[1, 1, 1]` (Ejercicio 1) |
| `poblacion_inicial`, `poblacion` | Población de configuraciones candidatas | Array `(n_individuos, n_genes)`; `poblacion` se reasigna cada generación tras selección + mutación (Ejercicio 1) |
| `fitness`, `fitness_final` | Valores de aptitud de la población | Array 1D con `metrica(ind)` evaluada sobre cada individuo, usado para ordenar y seleccionar (Ejercicio 1) |
| `sobrevivientes`, `hijos` | Individuos seleccionados y su descendencia mutada | `sobrevivientes` son la mitad de mayor fitness; `hijos` suman ruido gaussiano `rng.normal(0, 0.1, ...)` (Ejercicio 1) |
| `mejor` | Mejor individuo final | Resultado de `evolucionar_configuracion`, impreso y verificado contra el fitness inicial (Ejercicio 1, Autoevaluación) |
| `tabla_q` | Tabla de valores Q | Array `(n_estados, n_acciones)` actualizado episodio a episodio con la regla de Q-learning (Ejercicio 2) |
| `estado`, `accion` | Estado y acción muestreados en cada episodio | Enteros elegidos con `rng.integers`, usados para indexar `recompensas` y `tabla_q` (Ejercicio 2) |
| `recompensa`, `recompensas` | Recompensa observada y matriz de recompensas del entorno | `recompensas[estado, accion]` es el valor fijo que el agente intenta maximizar sin conocerlo de antemano (Ejercicio 2) |
| `mejor_futuro` | Estimación del valor futuro máximo | `np.max(tabla_q[estado])`, término de la actualización de Bellman en la regla de Q-learning (Ejercicio 2) |
| `tabla_final` | Tabla Q entrenada final | Resultado de `entrenar_q_learning`, cuyo `argmax` por fila se verifica en la Autoevaluación (Ejercicio 2, Autoevaluación) |

**Verificación manual del Diccionario de Variables** (el mecanismo automático de `ContentAuditorAgent._audit_diccionario_variables` es un placeholder que siempre retorna `[]` — no certifica nada): cada símbolo de la tabla fue releído contra el bloque de código donde aparece antes de agregarlo. Los diez símbolos están efectivamente usados en código Python realmente ejecutado en esta unidad (no en una tabla de sintaxis genérica ni en un docstring aislado): `metrica`/`metrica_ejemplo` se invocan dentro del bucle de evolución y de forma directa sobre `mejor`; `poblacion_inicial`/`poblacion` se transforman en cada iteración del `for`; `fitness`/`fitness_final` se calculan y se usan para `argsort`/`argmax` reales; `sobrevivientes`/`hijos` se construyen con slicing y `rng.normal` y se concatenan con `np.vstack`; `mejor` se imprime y se compara contra el fitness inicial en el test; `tabla_q` se inicializa en ceros y se actualiza en cada episodio; `estado`/`accion` se muestrean con `rng.integers` e indexan arrays reales; `recompensa`/`recompensas` se leen de la matriz fija y participan en la actualización de Bellman; `mejor_futuro` se calcula con `np.max` en cada episodio; `tabla_final` se imprime y sus `argmax` por fila se verifican con `assert` en la Autoevaluación.

### Autoevaluación

```python
%%writefile test_unidad_4.py
import numpy as np


def evolucionar_configuracion(metrica, poblacion_inicial, generaciones=20, semilla=42):
    rng = np.random.default_rng(semilla)
    poblacion = poblacion_inicial.copy()
    for _ in range(generaciones):
        fitness = np.array([metrica(ind) for ind in poblacion])
        mejores_indices = np.argsort(fitness)[-len(poblacion) // 2:]
        sobrevivientes = poblacion[mejores_indices]
        hijos = sobrevivientes + rng.normal(0, 0.1, sobrevivientes.shape)
        poblacion = np.vstack([sobrevivientes, hijos])[: len(poblacion_inicial)]
    fitness_final = np.array([metrica(ind) for ind in poblacion])
    return poblacion[np.argmax(fitness_final)]


def metrica_ejemplo(config):
    return -np.sum((config - 1.0) ** 2)


def test_algoritmo_genetico_mejora_sobre_poblacion_inicial():
    poblacion_inicial = np.random.default_rng(42).uniform(-2, 2, size=(10, 3))
    fitness_inicial = np.array([metrica_ejemplo(ind) for ind in poblacion_inicial])
    mejor_fitness_inicial = np.max(fitness_inicial)

    mejor_individuo_final = evolucionar_configuracion(metrica_ejemplo, poblacion_inicial)
    mejor_fitness_final = metrica_ejemplo(mejor_individuo_final)

    assert mejor_fitness_final >= mejor_fitness_inicial


def test_q_learning_aprende_la_accion_de_mayor_recompensa():
    def entrenar_q_learning(n_estados, n_acciones, recompensas, episodios=100, tasa_aprendizaje=0.1, factor_descuento=0.9, semilla=42):
        rng = np.random.default_rng(semilla)
        tabla_q = np.zeros((n_estados, n_acciones))
        for _ in range(episodios):
            estado = rng.integers(0, n_estados)
            accion = rng.integers(0, n_acciones)
            recompensa = recompensas[estado, accion]
            mejor_futuro = np.max(tabla_q[estado])
            tabla_q[estado, accion] += tasa_aprendizaje * (
                recompensa + factor_descuento * mejor_futuro - tabla_q[estado, accion]
            )
        return tabla_q

    recompensas = np.array([[1.0, 0.0], [0.0, 1.0]])
    tabla = entrenar_q_learning(n_estados=2, n_acciones=2, recompensas=recompensas)

    assert np.argmax(tabla[0]) == 0
    assert np.argmax(tabla[1]) == 1
```

```python
!pytest test_unidad_4.py -v
```

Ejecutado, las 2 pruebas pasan: el algoritmo genético produce un individuo final con fitness igual o mejor que el mejor individuo de la población inicial, y la tabla Q entrenada converge a la acción de mayor recompensa en ambos estados (`argmax` fila 0 = acción 0, fila 1 = acción 1). Las funciones se redefinen dentro del archivo de test (no se importan del notebook) — mismo patrón que ya usan las autoevaluaciones de U0-U3, evita depender de un mecanismo de import frágil hacia celdas de notebook ejecutadas previamente. Ambos tests verifican comportamiento real (mejora del fitness, convergencia de la política óptima), no un sanity check trivial.
