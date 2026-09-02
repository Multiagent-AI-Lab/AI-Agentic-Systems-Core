# GOVERNANCE.md

## 1. Propósito de este documento

Este archivo documenta el patrón pedagógico central que estructura el contenido de "Sistemas de IA Agénticos" y el estándar de densidad/calidad que cada sección de una unidad debe cumplir. No describe un sistema de agentes que genere contenido automáticamente — es una guía de convenciones para quien escribe o revisa contenido (sesión de trabajo con el profesor, con o sin asistencia de IA).

Este documento nace de una comparación directa (2026-09-02) entre este repo y sus dos hermanos ya maduros, LP y Probabilidad: las unidades U1-U4 de este repo cubren cada concepto en una sola pasada superficial (~340 líneas/unidad, 7-17 encabezados), mientras que LP (~1300 líneas/unidad, 40+ encabezados) y Probabilidad (~960 líneas/unidad) exploran cada concepto en capas — contexto de dominio, analogía didáctica, formalización, código, ejercicio. El contenido existente de este repo es correcto y verificado (ver El Ciclo del Agente en `CLAUDE.md`), pero delgado. Los mínimos de esta sección existen para cerrar esa brecha en cualquier trabajo de ampliación futuro, no para el contenido ya publicado retroactivamente sin que se decida explícitamente revisarlo.

## 2. El Ciclo del Agente

Ver `CLAUDE.md`, sección "Convenciones del contenido pedagógico", para la definición completa de las 6 fases (Selección de Arquitectura, Diseño, Implementación, Evaluación, Despliegue, Iteración) y su verificación por `ContentAuditorAgent`. Es el análogo, en este curso, del "Hilo de Oro" de LP y del "Ciclo de Verificación Triple" de Probabilidad — mismo principio de verificación en capas sucesivas, adaptado a ingeniería de sistemas de IA en vez de código imperativo o contenido matemático.

## 3. El Gold Standard de Densidad

Cada sección de contenido nuevo (un `##` o `###` que introduce un concepto no trivial) debe cumplir, salvo que se documente explícitamente por qué no aplica:

1. **Contexto de dominio antes del código**: al menos un párrafo que motive por qué el concepto importa en la práctica de sistemas de IA reales — no solo su definición técnica. LP antepone "Contexto Conceptual e Importancia en Ingeniería" a cada unidad; este curso debe hacer lo mismo antes de cualquier bloque de código nuevo.
2. **Al menos una analogía o comparación estructurada** para conceptos de arquitectura/diseño que no tengan una intuición obvia desde el código mismo (p. ej. framework de orquestación vs. protocolo de interoperabilidad, memoria de ventana deslizante vs. memoria episódica). No aplica a secciones puramente de implementación donde el código es autoexplicativo.
3. **Ejemplo de código ejecutado**, como ya se hace: bloque real, no pseudocódigo, con el resultado de ejecución citado en prosa inmediatamente después (patrón ya presente y correcto en todo el contenido existente — no se relaja).
4. **Al menos un ejercicio de práctica graduado por sección principal** (no solo el ejemplo de la sección y el cierre auto-referencial de la unidad) — algo que el estudiante resuelva variando el ejemplo dado, no solo lea. Ausente hoy en U1-U3; único precedente parcial en U4 (2 ejercicios, por ser la unidad completa, no por sección).
5. **El Ciclo del Agente completo** (sección 2) — ya exigido y verificado automáticamente, sin cambios.
6. **Diccionario de Variables verificado manualmente** contra código realmente ejecutado — ya exigido, ver `CLAUDE.md` (el auditor automático de esta dimensión es un placeholder, no un sustituto de la revisión manual).
7. **Vigencia de la información**: cualquier referencia a versiones de librerías, frameworks, o el panorama de herramientas de un dominio (como el panorama de frameworks 2026 de U3) debe verificarse contra el estado real al momento de escribir o revisar — no asumir que información previa sigue vigente sin confirmarlo.

## 4. Densidad mínima de referencia

No se fija un mínimo de palabras rígido (a diferencia de Probabilidad, cuyo dominio matemático se presta a un umbral cuantitativo como "≥800 palabras de teoría") porque el contenido de este curso mezcla teoría de arquitectura, código, y ejercicios en proporciones que varían por tema. En su lugar, la referencia de densidad es comparativa: una unidad nueva o ampliada de este repo debe aproximarse al orden de magnitud de LP/Probabilidad (800-1300 líneas de Markdown, 30-45 encabezados) antes de considerarse completa — una unidad que queda por debajo de la mitad de ese rango es candidata a revisión de ampliación, no a aprobación directa.

## 5. Verificación de Símbolos en el Diccionario de Variables

Cada entrada del Diccionario de Variables debe corresponder a un símbolo o variable usada en un ejemplo REAL Y EJECUTADO de la propia unidad — una tabla de sintaxis genérica, un docstring en prosa, o una mención aislada sin ejemplo aplicado no cuentan como uso verificado. Antes de agregar o aprobar una entrada, releer el bloque de código que la usa. Mismo criterio que LP y Probabilidad; ya aplicado correctamente en el contenido existente de este repo.

## 6. Alcance de este documento

Este documento se aplica a trabajo de ampliación o creación de contenido pedagógico futuro. No implica que las unidades U1-U4 actuales estén "incompletas" en el sentido de tener errores — están verificadas y correctas dentro de su alcance actual. Implica que, cuando se decida ampliar su densidad (fuera del alcance de este documento en sí, que es solo el estándar — la decisión y ejecución de la ampliación es un trabajo aparte, documentado en su propia spec de `docs/superpowers/specs/`), estos son los criterios contra los que se mide "completo".
