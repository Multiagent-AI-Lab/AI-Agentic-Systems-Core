# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Material del curso genérico "Sistemas de IA Agénticos" (UCEMICH, Ingeniería en IA y Nanotecnología). Repo hermano de [Programming-Logic-Agentic-AI-Development](https://github.com/Multiagent-AI-Lab/Programming-Logic-Agentic-AI-Development) (LP), [Probability-Statistics-Agentic-AI-Core](https://github.com/Multiagent-AI-Lab/Probability-Statistics-Agentic-AI-Core) (Probabilidad) y [Antigravity-Nano-Research-Multiagentic-Core](https://github.com/Multiagent-AI-Lab/Antigravity-Nano-Research-Multiagentic-Core) (Antigravity-Nano). A diferencia de los otros tres, este repo no tiene dominio propio: cubre los fundamentos genéricos de IA/ML/Sistemas Multi-Agente que antes vivían mezclados con contenido de nanotecnología en Antigravity-Nano, y Antigravity-Nano ahora depende de él como prerequisito curricular. Se imparte de forma simultánea a Antigravity-Nano en la UCEMICH — alguien puede cursar solo este repo (sistemas multiagente aplicables a cualquier dominio), pero nadie cursa Antigravity-Nano sin haber visto este primero.

5 unidades en `lecciones/` (U0-U4), un sistema de agentes Python en `src/multiagent_core/`, y un pipeline que convierte esas unidades en Jupyter Notebooks. `docs/superpowers/specs/` documenta las decisiones de diseño de cada sub-proyecto (separación del contenido de Antigravity-Nano, infraestructura de agentes, contenido pedagógico, migración de la práctica de Sistemas Multi-Agente) — revisar antes de proponer algo que pueda solaparse.

## Comandos esenciales

```bash
# Activar entorno (conda, propio de este repo — independiente del ia_nano de Antigravity-Nano)
conda activate sys_agents

# Correr toda la suite de tests
pytest tests/ -v --tb=short

# Correr un solo archivo de test
pytest tests/test_tutor_agent.py -v

# Correr un solo test
pytest tests/test_tutor_agent.py::TestAsk::test_construye_prompt_con_contexto_y_pregunta -v

# Formatear/lint (en ese orden) antes de dar por terminado cualquier cambio en src/
python -m isort src/multiagent_core/<archivo>.py
python -m black src/multiagent_core/<archivo>.py
python -m ruff check src/multiagent_core/<archivo>.py

# Regenerar los notebooks tras editar cualquier lecciones/UNIDAD_*.md
python convert_to_notebooks.py
```

Requiere Node.js instalado (`npx` en el PATH) — los diagramas Mermaid se renderizan a SVG vía `@mermaid-js/mermaid-cli`; el script verifica esto al inicio y falla con instrucciones claras si falta.

`pyproject.toml` declara `src*`/`multiagent_core*` en `[tool.setuptools.packages.find]` y trae `pytest`/`pytest-asyncio`/`pytest-cov`/`pytest-mock`/`ruff`/`black`/`isort` en el extra `dev`. Los imports en el código y en los tests son siempre absolutos desde la raíz: `from src.multiagent_core.tutor_agent import TutorAgent`.

**`conda run -n sys_agents python -c "..."` falla con un `AssertionError` de conda si el script pasado con `-c` contiene saltos de línea** (bug conocido del entorno, no del proyecto) — escribir el script a un archivo temporal y ejecutar ese archivo en su lugar, nunca `-c` multilínea.

## Arquitectura del sistema de agentes

15 módulos en `src/multiagent_core/`, en dos capas:

**Agentes originales** (heurísticos y sin LLM salvo `TutorAgent`, migrados/replicados desde LP con el mismo diseño):

- **`NotebookCompilerAgent`** (`notebook_compiler_agent.py`) — parsea un `UNIDAD_*.md` línea por línea (vía `_scan_fence_segments`/`extract_fenced_blocks` en `_fence_utils.py`, compartido con `ContentAuditorAgent`) y genera el `.ipynb` correspondiente, preservando el orden real de aparición de texto y código del documento fuente. Reconoce fences de longitud variable (3+ backticks); el fence de cierre debe coincidir exactamente en longitud con el de apertura. Usa `MathAgent` (clase interna) para traducir símbolos Unicode matemáticos a LaTeX, y `FlowchartAgent` para inyectar diagramas Mermaid automáticos cuando detecta una función de más de 5 líneas.
- **`FlowchartAgent`** — AST de Python → diagrama Mermaid (`graph TD`).
- **`PseudocodeAgent`** — traduce entre pseudocódigo, Mermaid, y esqueletos Python.
- **`CodeAuditorAgent`** — análisis estático (AST + regex, `_security_patterns.py` compartido con `ContentAuditorAgent` y `SafetyGateAgent`) de estilo PEP8 y seguridad OWASP; también ejecuta pytest vía subprocess (`run_pytest`).
- **`EvaluatorAgent`** — califica código contra los criterios de `RUBRICA_GENERAL.md`, reutilizando `CodeAuditorAgent` internamente. El mensaje de proceso nombra explícitamente las 6 fases del Ciclo del Agente (ver abajo).
- **`OrchestratorAgent`** — coordina Auditor + Flowchart + Evaluator en un reporte Markdown único.
- **`SafetyGateAgent`** — guardrail heurístico contra prompt injection reflejado en outputs de agentes (detecta patrones de instrucciones de sistema reflejadas, API keys filtradas, formato de salida inválido). Detecta explícitamente **solo** ese vector — no memory poisoning ni tool misuse (ver Seguridad de Agentes en U3).
- **`TutorAgent`** — el único agente con LLM: RAG semántico con ChromaDB sobre los `.md` de `lecciones/` (indexados por sección, citando fuente exacta) + Gemini para responder, con debugger socrático y memoria episódica local. Usa el SDK `google.genai` (no el deprecado `google.generativeai`); el `Client` se crea de forma perezosa dentro de `ask()`, leyendo `GEMINI_API_KEY` de `os.environ` en ese momento. La colección de ChromaDB usa `SentenceTransformerEmbeddingFunction` con el modelo multilingüe `paraphrase-multilingual-MiniLM-L12-v2`. `chroma_path` y `memory_path` son inyectables en el constructor para aislar tests con `tmp_path` — cualquier ruta de archivo/directorio nueva que se agregue a un agente debe seguir ese mismo patrón, nunca hardcodear rutas a nivel de módulo (ver el fix de `episodic_retriever.py` abajo como ejemplo del antipatrón corregido).
- **`ContentAuditorAgent`** — heurístico, sin LLM: audita el contenido pedagógico de un `UNIDAD_*.md`. `_audit_pedagogico` verifica que las 6 fases del Ciclo del Agente (`FASES_CICLO_DEL_AGENTE`, ver abajo) estén presentes como sub-encabezados `###` exactos, excluyendo unidades marcadas como "Línea de Investigación" (`_ARCHIVO_EXENTO_DEL_CICLO`, hoy solo U4). `audit_all_units(course_dir)` itera todas las `UNIDAD_*.md` y agrega un reporte consolidado. **`_audit_diccionario_variables` es un placeholder que siempre retorna `[]`** — no verifica nada real; cualquier verificación de esa dimensión debe declararse explícitamente como manual, nunca presentarse como certificación automática del auditor.
- **`PdfIndexer`** (`pdf_indexer.py`) — indexación de PDFs para RAG (usado por `TutorAgent` si se agregan fuentes bibliográficas).

**Módulos migrados desde Antigravity-Nano** (`src/multiagent_core/skills/`, Sub-proyecto de migración de práctica U5→U3, sin LLM, solo stdlib salvo dependencias declaradas en el extra `practica-u3`):

- `agent_warmup/context_loader.py` — inyecta contexto de dominio en agentes LLM (5 dominios predefinidos + custom).
- `routing/task_classifier.py` — clasifica tareas por reglas de palabras clave y recomienda agente.
- `evaluation/output_scorer.py` — scoring heurístico de outputs de agentes.
- `memory/episodic_retriever.py` — memoria episódica con backend Mem0 (cloud) y fallback JSON local; `store_path` es un parámetro inyectable resuelto en tiempo de llamada (`_default_store_path()`), nunca una constante de módulo — corregido durante la migración porque el original de Antigravity-Nano sí lo tenía hardcodeado, haciendo imposible aislar tests con `tmp_path`.
- `observability/trace_annotator.py` — decorador de tracing con logging de fallback.
- `apis/token_budget_guard.py` — control de presupuesto de tokens por modelo.
- `apis/github_skill_loader.py` — descarga y registro seguro de skills desde GitHub con verificación SHA-256 y allowlist de dominio (anti-SSRF).
- `registry.py` — registro versionado (`SKILL_REGISTRY`) de los 7 módulos anteriores, con alias `@latest` automático.

**Pipeline de contenido**: `lecciones/UNIDAD_*.md` (fuente de verdad) → `convert_to_notebooks.py` → `NotebookCompilerAgent.compile()` → `notebooks/*.ipynb`. Los notebooks generados nunca se editan a mano; cualquier cambio de contenido va al `.md` en `lecciones/` y se regenera. `notebooks/practica_u3/` es un caso aparte: 8 notebooks + `u3_08_api.py` de **autoría directa** (migrados desde la Unidad 5 de Antigravity-Nano, no derivados de ningún `.md`) — `convert_to_notebooks.py` no los toca ni los lista, y no deben regenerarse con el pipeline normal.

## Convenciones del contenido pedagógico

- **El Ciclo del Agente**: cada concepto nuevo del curso recorre 6 fases, en este orden exacto y con estos nombres literales (verificados por `ContentAuditorAgent` como sub-encabezados `### <Fase>` bajo `## 🔄 El Ciclo del Agente`): **Selección de Arquitectura, Diseño, Implementación, Evaluación, Despliegue, Iteración**. Es el equivalente de este curso al "Hilo de Oro" (Pseudocódigo→Mermaid→Python→pytest) de LP y al "Ciclo de Verificación Triple" de Probabilidad — mismo principio de verificación en capas sucesivas, adaptado a ingeniería de sistemas de IA. Presente completo en U1-U3; **exento explícitamente en U4** (marcada "Línea de Investigación", contenido de frontera sin el mismo nivel de estandarización de producción).
- **Anatomía del Agente** (U1): vocabulario base que precede al Ciclo — `Agente = Modelo + Harness`, con el Harness proveyendo Tools (APIs/ejecutores/navegadores), Memory (historial/contexto/estado), Guardrails (reglas de seguridad/límites/verificaciones). Se referencia hacia atrás en unidades posteriores.
- **Diccionario de Variables**: cada unidad cierra con una sección `### Diccionario de Variables` (símbolo/nombre + descripción). Cada entrada debe estar usada en un ejemplo de código **realmente ejecutado** dentro de la propia unidad — una tabla de sintaxis genérica, un docstring en prosa, o una mención aislada no cuentan como uso real. `ContentAuditorAgent._audit_diccionario_variables` NO verifica esto automáticamente (placeholder, ver arriba) — se verifica manualmente releyendo el bloque de código citado antes de agregar o aprobar una entrada.
- **Cierre auto-referencial**: cada unidad de U1-U3 cierra su sección de contenido principal con un ejemplo que importa y usa (nunca simula) un agente real de `src/multiagent_core/` — conectando el contenido teórico de la unidad con la infraestructura de software que el propio curso construye.
- **Contexto de dominio**: los ejemplos de código usan datasets/problemas reales de ML/IA (Iris, California Housing, Wine, optimización bayesiana, detección de anomalías) — no ejemplos de nanotecnología (ese es el rol de Antigravity-Nano, que depende de este repo, no al revés) ni ejemplos genéricos de programación sin contexto.
- **Notebooks de práctica de U3** (`notebooks/practica_u3/`): profundizan con implementación real (LangGraph, CrewAI, Google ADK/A2A, RAG, GraphRAG, multimodal, proyecto integrador) lo que la sección teórica de `UNIDAD_3_SISTEMAS_MULTI_AGENTE.md` cubre a nivel conceptual. Requieren el extra `pip install -e ".[practica-u3]"`, con versiones de dependencias ancladas a su validación original (marzo 2026 en el curso de origen) — no se actualizan a versiones más nuevas sin verificar antes que los notebooks siguen funcionando. `langchain-google-genai==2.0.10` (en vez de `2.1.4`) es una deuda conocida documentada en la unidad: resuelve sin conflicto de dependencias pero sin validación end-to-end contra la API real.
- **Densidad de contenido — deuda pedagógica abierta**: comparado con LP (~1300 líneas/unidad, 40+ encabezados) y Probabilidad (~960 líneas/unidad), las unidades U1-U4 de este repo son notablemente más delgadas (~340 líneas/unidad, 7-17 encabezados) y carecen de las analogías didácticas extensas, la exploración en capas del contexto de dominio, y los ejercicios de práctica intermedios que sí tienen los otros dos repos hermanos. El contenido existente es correcto y verificado (Ciclo del Agente completo, tests que pasan, cierres auto-referenciales reales) pero cubre cada tema en una sola pasada en vez de la profundidad esperada — pendiente de una revisión de ampliación de contenido, no de un fix de corrección.

## Flujo de trabajo esperado para cambios grandes

Este proyecto usa el flujo `superpowers:brainstorming` → spec en `docs/superpowers/specs/` → `superpowers:writing-plans` → plan en `docs/superpowers/plans/` para cualquier trabajo no trivial (nuevo agente, cambio de arquitectura, ampliación de contenido pedagógico). Los specs ya escritos documentan decisiones de diseño y su razonamiento — revisarlos antes de proponer algo que pueda solaparse.

TDD estricto en todo el código de `src/multiagent_core/`: test que falla primero, luego implementación mínima.
