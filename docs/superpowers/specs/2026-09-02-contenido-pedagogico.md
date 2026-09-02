# Sub-proyecto 3: Contenido Pedagógico — Spec

## Contexto

Sub-proyecto 3 de la secuencia definida en `2026-09-01-separacion-y-nivelacion-antigravity-nano.md`. El Sub-proyecto 2 (infraestructura de agentes) ya está completo: los 8 agentes en `src/multiagent_core/` funcionan sobre contenido de prueba mínimo (`lecciones/UNIDAD_0_PRUEBA.md`). Este sub-proyecto escribe el contenido pedagógico real que reemplaza ese placeholder, y ajusta el código de dos agentes que dependían de un nombre de patrón pedagógico aún no decidido.

`AI-Agentic-Systems-Core` **no es un curso optativo**: se imparte de forma **simultánea** con Antigravity-Nano-Research-Multiagentic-Core en UCEMICH (ambos son parte regular del programa, ninguno es prerequisito temporal del otro), y es además **autónomo** — alguien externo a UCEMICH puede cursar únicamente este repo sin necesidad de Antigravity-Nano. Esto tiene una consecuencia de diseño estricta: el contenido de este repo no asume que el alumno cursa Antigravity-Nano en paralelo, y **no contiene ninguna referencia a nanotecnología** — el objetivo explícito es dejar al alumno capaz de aplicar sistemas multiagente a cualquier dominio después (nanotecnología en el otro repo, u otro dominio cualquiera), no enseñar nanotecnología aquí.

## Decisiones de diseño (brainstorming, confirmadas por el usuario)

1. **Audiencia real**: no es el mismo alumno de primer semestre de LP/Probabilidad. Es alguien que ya tiene base matemática (álgebra lineal, cálculo, probabilidad) y quiere aprender a construir sistemas multiagente de calidad. Esto corrige una suposición inicial del brainstorming (que asumía audiencia de primer semestre) — la spec madre no fijaba esto explícitamente, así que se decide aquí.
2. **Unidad 0 = contenido real de EMALCA, adaptado, no solo referencia**. Se descartó la alternativa de tratar EMALCA como lectura complementaria aparte: dado que la audiencia real ya tiene base matemática, el desajuste de nivel que motivó esa alternativa no aplica. U0 integra el contenido real de EMALCA (`C:\Users\ljyud\Desktop\Fundamentos Matemáticos de la Inteligencia Artificial\notas_fundamentos_matematicos_ia.md`) adaptado al estándar del resto del curso (Diccionario de Variables, ejemplos ejecutados, autoevaluación) — no es una versión simplificada para principiantes, ya que la audiencia no lo necesita.
3. **Patrón pedagógico central: "El Ciclo del Agente"**, 6 fases — **Selección de Arquitectura → Diseño → Implementación → Evaluación → Despliegue → Iteración**. Enriquecido explícitamente más allá del ciclo académico simple (diseño→implementación→evaluación) para reflejar cómo se construyen sistemas multiagente reales en producción: la fase de Selección de Arquitectura cubre la decisión de un agente vs. varios y qué framework usar según el problema (LangGraph/CrewAI/ADK/etc.); la fase de Despliegue cubre costos, latencia y observabilidad — dimensiones ausentes del Hilo de Oro de LP y del Ciclo de Verificación Triple de Probabilidad, propias de este dominio.
4. **Política de IA: permitida con verificación crítica desde el inicio** (mismo enfoque que Probabilidad, no progresivo como LP). Justificación: este curso es *sobre* construir con IA — restringir su uso en las primeras unidades no tiene sentido pedagógico aquí, a diferencia de LP (curso de fundamentos de programación donde restringir IA inicialmente sí tiene sentido).
5. **Dominio de ejemplos de código en U1/U2**: datasets públicos clásicos de ML (Iris, MNIST, California housing — reconocibles, ya usados en notebooks existentes de Antigravity-Nano) como base de cada unidad, más un ejemplo de cierre auto-referencial que aplica el concepto a los propios agentes del curso (ej. clasificar tipos de hallazgo de `CodeAuditorAgent`, o predecir cuántos hallazgos tendrá un `ContentAuditorAgent` según features del contenido auditado) — conecta la teoría de ML con el resto del sistema sin depender de un dominio de aplicación ajeno al curso. Sin datos sintéticos genéricos salvo que un concepto puntual los requiera (ej. visualizar un límite de decisión).
6. **`AI-Agentic-Systems-Core` es curricularmente autónomo**: aunque se imparte simultáneo con Antigravity-Nano, no se asume que el alumno lo curse. Ningún ejemplo, ejercicio o referencia de este repo depende de contenido de Antigravity-Nano ni usa nanotecnología como dominio. La relación entre ambos repos (que Antigravity-Nano cite este repo como prerequisito curricular en sus propias unidades U3/U4/U5/U6, per la spec madre) es unidireccional: Antigravity-Nano depende de este repo, no al revés.

## Cambio de código requerido (no solo contenido)

El Sub-proyecto 2 dejó dos agentes con un placeholder textual (`<PATRÓN_PEDAGÓGICO>`) porque el nombre del patrón no estaba decidido, pero la **lógica** de verificación asumía las 4 fases del Hilo de Oro de LP (`pseudocodigo`, `mermaid`, `python`, `pytest`), que no corresponden a "El Ciclo del Agente". Esto no es un simple find-and-replace de texto — requiere TDD real sobre la nueva lógica:

- **`ContentAuditorAgent._audit_pedagogico`** (`src/multiagent_core/content_auditor_agent.py`): hoy verifica `{"pseudocodigo", "mermaid", "python", "pytest"} ⊆ idiomas_presentes` (bloques con fences de esos lenguajes). Debe reescribirse para verificar que las 6 fases de "El Ciclo del Agente" estén presentes y citadas en la unidad — el mecanismo de detección exacto (¿fences con nombres de fase? ¿encabezados `##` con el nombre de cada fase? ¿ambos?) se decide en el plan de implementación, pero debe ser una heurística real y verificable, no solo cambiar el texto del mensaje de hallazgo.
- **`EvaluatorAgent._evaluar_proceso`** (`src/multiagent_core/evaluator_agent.py`): hoy retorna siempre `"En desarrollo"` con un mensaje genérico sobre "el ciclo completo del patrón pedagógico". El mensaje se actualiza para nombrar "El Ciclo del Agente" y sus 6 fases explícitamente en la retroalimentación al estudiante — la limitación de no poder verificar el ciclo completo automáticamente a partir del código fuente solo (ya documentada y aceptada en el Sub-proyecto 2) se mantiene, no se intenta resolver aquí.
- Ambos cambios llevan su propio ciclo TDD (test que falla primero, luego implementación) — no son un simple "buscar y reemplazar texto", son un cambio de comportamiento verificable.

## Mapeo de contenido (fuente → unidad de este repo)

| Unidad | Fuente | Tratamiento |
|---|---|---|
| U0 — Fundamentos Matemáticos | EMALCA, Capítulos 0-8 (`notas_fundamentos_matematicos_ia.md`) | Adaptado al estándar del curso: Diccionario de Variables por capítulo, ejemplos ejecutados en Python (no solo fórmulas), autoevaluación. Mantiene el rigor formal de EMALCA (la audiencia lo soporta) pero agrega el andamiaje pedagógico que el original no tiene (fue escrito para una charla, no para un curso con autoevaluación). |
| U1 — ML Fundamentals | `unit_03_ml_nanomaterials/` de Antigravity-Nano, ML clásico y redes neuronales generalizados | Se quita todo dato/ejemplo de nanopartículas; se reemplaza por datasets clásicos (Iris/MNIST/California housing) + cierre auto-referencial. |
| U2 — IA Aplicada Genérica | `unit_04_applied_ai/` de Antigravity-Nano, optimización bayesiana y detección de anomalías generalizadas | Se quita SEM/espectroscopía/óxidos; mismo patrón de dominio que U1. |
| U3 — Sistemas Multi-Agente | `unit_05_multi_agent_sys/` de Antigravity-Nano (LangChain/LangGraph, CrewAI, Google ADK/A2A, RAG/GraphRAG, multimodal) completa, fusionada con Capítulos 9 y 10-bis de EMALCA (Teoría de Juegos, Ingeniería de Agentes) | Ya es 100% genérica en la fuente (confirmado en la sesión anterior: sin ninguna referencia a nanotecnología) — se traslada y se enriquece con la fundamentación matemática formal de EMALCA (teoría de juegos para coordinación multiagente) sin duplicar contenido ya cubierto. Es la unidad donde "El Ciclo del Agente" se enseña explícitamente por primera vez con todo su rigor (Selección de Arquitectura y Despliegue tienen más peso aquí que en U1/U2). |

## Estándar de calidad (heredado del Sub-proyecto 2, sin cambios)

- TDD estricto para los 2 cambios de código (`_audit_pedagogico`, `_evaluar_proceso`).
- `ruff check` limpio.
- Diccionario de Variables con el criterio estricto ya aplicado en LP/Probabilidad: un símbolo cuenta como usado solo si aparece en código realmente ejecutado en la propia unidad, nunca en una tabla genérica o docstring en prosa.
- Cada unidad cita "El Ciclo del Agente" de forma explícita, con al menos las 6 fases nombradas y aplicadas al ejemplo central de la unidad.
- Contexto de dominio: datasets clásicos de ML + ejemplos auto-referenciales a los agentes del curso, nunca nanotecnología.
- Autoevaluación ejecutable (pytest/ipytest) en cada unidad, mismo patrón que LP/Probabilidad.
- Ejemplos de código en U0 (fórmulas de EMALCA traducidas a código) verificados con ejecución real antes de publicarse — mismo principio que ya evitó errores reales en Probabilidad (salida documentada que no coincidía con ejecución real).

## Fuera de alcance de este sub-proyecto

- Migración de estos agentes/contenido a Antigravity-Nano — Sub-proyecto 4.
- Reescritura de U3/U4/U6 de Antigravity-Nano para citar este repo como prerequisito — Sub-proyecto 6.
- Cualquier referencia a nanotecnología en este repo — explícitamente prohibido por diseño (ver Decisión 6).
- Traducción de contenido a otros idiomas.
