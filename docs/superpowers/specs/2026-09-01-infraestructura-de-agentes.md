# Sub-proyecto 2: Infraestructura de Agentes — Spec

## Contexto

Sub-proyecto 2 de la secuencia definida en `2026-09-01-separacion-y-nivelacion-antigravity-nano.md`. El Sub-proyecto 1 (repo base, `pyproject.toml`, entorno conda `sys_agents`, CI) ya está completo y publicado. Este sub-proyecto construye los 8 agentes de software en `src/multiagent_core/`, réplica del patrón ya probado y calificado 98/100 en Programming-Logic-Agentic-AI-Development (LP) y Probability-Statistics-Agentic-AI-Core (Probabilidad).

## Decisiones de diseño (brainstorming, confirmadas por el usuario)

1. **`SafetyGateAgent` es genérico en este repo, no específico de nanotecnología/química.** La spec original de separación decía "idénticos en ambos repos" para los 8 agentes, pero eso contradecía la razón de ser de la separación: `stability_guardian.py`/RDKit son herramientas de química, exactamente lo que este repo debe NO tener. Resolución: aquí `SafetyGateAgent` valida guardrails genéricos de outputs de agentes (prompt injection reflejado, formato esperado, filtrado de secretos) y reconecta la idea de retroalimentación socrática de forma genérica. En Antigravity-Nano (Sub-proyecto 4), se extiende con las validaciones químicas reales (RDKit, `stability_guardian` rescatado) — mismo nombre de clase, responsabilidad base compartida, especialización distinta por repo.
2. **8 agentes heurísticos salvo `TutorAgent`, que usa LLM (Gemini).** Se evaluó explícitamente la alternativa de dar LLM a más agentes ("sistemas multiagente de verdad") y se descartó: los 7 agentes de auditoría/compilación/evaluación necesitan ser deterministas y verificables para servir como gate de calidad — un auditor de seguridad no determinista es un auditor roto. "Sistemas multiagente reales" (LangGraph/CrewAI/ADK, LLMs orquestando LLMs) es tema de **contenido pedagógico** del Sub-proyecto 3 (lo que el curso enseña a construir), no un cambio de arquitectura de estos 8 agentes de tooling.
3. **`TutorAgent` se construye ahora, sin contenido pedagógico real todavía.** El Sub-proyecto 3 llena `lecciones/` y `bibliografia/` con contenido real; este sub-proyecto construye el agente completo (RAG con ChromaDB + Gemini) probado con TDD sobre una carpeta `lecciones/` mínima/de prueba. El código del agente no cambia cuando el contenido real llegue.
4. **`TutorAgent` usa Gemini** (`google.genai`, `GEMINI_API_KEY`), mismo patrón que LP/Probabilidad — no un proveedor distinto.
5. **`ContentAuditorAgent` usa un set de 6 dimensiones propio de este repo** (no las 5 de LP ni las 9 de Probabilidad, aunque el patrón general — heurístico, sin LLM, reutiliza `CodeAuditorAgent` — sí se hereda):
   1. LaTeX bien formado (matemáticas de ML/optimización).
   2. Patrón pedagógico propio completo (Hilo de Oro / Ciclo de Verificación Triple / equivalente — su nombre se define en el Sub-proyecto 3; esta dimensión audita que el ciclo esté citado y completo, sea cual sea su nombre final).
   3. Código de ejemplo con docstrings, type hints, sin riesgos OWASP.
   4. Reproducibilidad de agentes: seeds fijados, configuración explícita de modelo/temperatura en cualquier ejemplo con LLM, ninguna clave de API hardcodeada.
   5. Diccionario de Variables verificado contra código realmente ejecutado en la propia unidad (mismo criterio estricto que causó rechazos reales de revisión en Probabilidad: nunca un símbolo listado que solo aparece en una tabla genérica o docstring en prosa).
   6. Invariantes estructurales: fences de código balanceados, celdas de autoevaluación consistentes.

## Arquitectura

Mismo patrón exacto que LP: cada agente es una clase en su propio archivo bajo `src/multiagent_core/`, sin LLM salvo `TutorAgent`, rutas de archivo/directorio siempre inyectables en el constructor (nunca hardcodeadas) para poder aislar tests con `tmp_path`. Las utilidades reales rescatables de `external_skills/` de Antigravity-Nano (`context_loader.py`, `github_skill_loader.py`, `token_budget_guard.py`, `output_scorer.py`, `episodic_retriever.py`, `graph_memory.py`, `basis_set_architect.py`, `stability_guardian.py`, `trace_annotator.py`, `task_classifier.py`) se copian a este repo bajo `src/multiagent_core/_vendored/` si el agente que las necesita ya está en el orden de construcción de este sub-proyecto; si ninguno las necesita todavía, su copia se pospone al sub-proyecto de contenido/migración que sí las use — evita importar código sin un consumidor real (YAGNI).

## Orden de construcción (cada tarea depende de la anterior salvo que se indique independiente)

1. `FlowchartAgent` — independiente.
2. `PseudocodeAgent` — independiente.
3. `CodeAuditorAgent` — independiente.
4. `ContentAuditorAgent` — depende de `CodeAuditorAgent`.
5. `NotebookCompilerAgent` (+ `MathAgent` interno) — depende de `FlowchartAgent`.
6. `EvaluatorAgent` — depende de `CodeAuditorAgent`.
7. `SafetyGateAgent` — independiente.
8. `OrchestratorAgent` — depende de `CodeAuditorAgent`, `FlowchartAgent`, `EvaluatorAgent`, `SafetyGateAgent`.
9. `TutorAgent` — independiente de los anteriores; depende de ChromaDB + Gemini.

## Interfaces (contratos exactos — el plan de implementación no debe inventar firmas nuevas)

Todas las clases en `src/multiagent_core/<snake_case>.py`, imports absolutos desde raíz (`from src.multiagent_core.<agente> import <Clase>`).

- **`FlowchartAgent`**: `__init__() -> None`; `build_mermaid_flowchart(code_source: str) -> str`.
- **`PseudocodeAgent`**: `__init__() -> None`; `pseudocode_to_mermaid(pseudocode: str) -> str`; `python_to_pseudocode(code_source: str) -> str`; `pseudocode_to_python_skeleton(pseudocode: str) -> str`.
- **`CodeAuditorAgent`**: `__init__() -> None`; `audit_style(code: str) -> list[str]`; `audit_security(code: str) -> list[str]`; `run_pytest(test_file_path: Path) -> dict[str, Any]`.
- **`ContentAuditorAgent`**: `__init__() -> None`; `audit_unit(md_path: Path) -> dict[str, Any]` — retorna hallazgos por cada una de las 6 dimensiones fijadas arriba, más un resumen (`passed: bool`, `score: float` o estructura equivalente, a definir en el plan siguiendo el estilo de retorno de LP).
- **`NotebookCompilerAgent`**: `__init__(mermaid_renderer: MermaidRenderer | None = None, asset_base_url: str = ...) -> None`; `compile(md_filepath: Path, output_dir: Path) -> Path`. `MathAgent` como clase interna del mismo archivo (`process_latex(text: str) -> str`), igual que en LP.
- **`EvaluatorAgent`**: `__init__() -> None`; `evaluar(student_code: str, test_file_path: Path | None) -> dict[str, Any]` contra 4 criterios de una rúbrica genérica propia de este repo (el contenido exacto de la rúbrica se redacta en el Sub-proyecto 3; esta tarea construye el motor de evaluación con criterios placeholder testeables, análogos en forma a los 4 de LP: corrección lógica, proceso, calidad de código, reproducibilidad).
- **`SafetyGateAgent`** (nuevo): `__init__() -> None`; `check_output(agent_output: str) -> dict[str, Any]` — retorna hallazgos de guardrails genéricos: patrones de prompt injection reflejados en el output, ausencia de secretos/claves filtradas, formato esperado no violado.
- **`OrchestratorAgent`**: `__init__() -> None`; `run(student_code: str, md_path: Path | None = None, test_file_path: Path | None = None) -> str` — reporte Markdown único, coordinando `CodeAuditorAgent`, `FlowchartAgent`, `EvaluatorAgent`, `SafetyGateAgent` (y `ContentAuditorAgent` cuando `md_path` se provee).
- **`TutorAgent`**: `__init__(course_dir: Path, chroma_path: Path | None = None, memory_path: Path | None = None, bibliografia_dir: Path | None = None) -> None` — construye el índice ChromaDB de forma eager en el constructor (`_build_index()`), mismo patrón que LP; `ask(question: str) -> str` — crea el `Client` de Gemini de forma perezosa dentro de este método, leyendo `GEMINI_API_KEY` de `os.environ` en ese momento (nunca en `__init__` ni a nivel de módulo), para que el agente sea instanciable sin la key (solo `ask()` retorna un mensaje de error controlado). Sin integración a Materials Project (eso es específico de Antigravity-Nano) ni a bibliografía de nanotecnología — la carpeta `bibliografia_dir` es genérica para PDFs de IA/ML/Sistemas Multi-Agente.

## Estándar de calidad

- TDD estricto para los 8 agentes, sin excepción — test que falla primero, luego implementación mínima, sin importar si el agente envuelve una utilidad rescatada o no.
- `ruff check` limpio en cada commit y en CI (ya configurado en Sub-proyecto 1).
- Rutas de archivo/directorio siempre inyectables en constructores — nunca hardcodeadas — para aislar tests con `tmp_path`, igual que TutorAgent de LP.
- Ningún agente hace red real en sus tests (Gemini, Crossref, cualquier API externa se mockea en tests unitarios; una prueba de integración real con Gemini, si se incluye, se marca explícitamente y no corre en CI sin `GEMINI_API_KEY`).

## Fuera de alcance de este sub-proyecto

- Contenido pedagógico real (`lecciones/*.md`) — Sub-proyecto 3.
- Nombre final del patrón pedagógico central (Hilo de Oro / equivalente propio) — Sub-proyecto 3.
- Redacción final de la rúbrica de 4 criterios de `EvaluatorAgent` — Sub-proyecto 3 (aquí se construye el motor con criterios placeholder testeables).
- Migración/adaptación de estos agentes a Antigravity-Nano, incluyendo la especialización química de `SafetyGateAgent` — Sub-proyecto 4.
- Integración con Materials Project API — específica de Antigravity-Nano, no de este repo.
