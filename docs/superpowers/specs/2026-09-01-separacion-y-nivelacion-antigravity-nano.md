# Separación y Nivelación de Antigravity-Nano — Spec

## Contexto y motivación

Antigravity-Nano-Research-Multiagentic-Core es el tercero de tres repos hermanos educativos del mismo profesor (UCEMICH), junto con Programming-Logic-Agentic-AI-Development (LP) y Probability-Statistics-Agentic-AI-Core (Probabilidad). Un diagnóstico técnico y pedagógico repetido a lo largo de varias sesiones (ver memoria `project_three_repos_diagnostico_2026_08_25.md`) encontró que Antigravity-Nano está muy por detrás de sus dos hermanos en infraestructura real:

- **0 agentes de software reales**: `external_skills/` contiene 21 módulos sueltos sin ningún orquestador; el "Consejo de 7 Expertos" documentado en `GOVERNANCE.md`/`README.md` (@Architect, @Scientist, @Engineer, @Safety_Gate, @Analyst, @Librarian, @QA) es un flujo de prompts para *generar* contenido, no software ejecutable — de hecho, 3 de los skills que el README les atribuye (`descriptor_miner`, `data-storytelling`, `senior-architect`) no existen como código en absoluto.
- **RAG mock**: `external_skills/orchestration/librarian_rag.py` es un diccionario Python hardcodeado de 4 compuestos (19 líneas), con el docstring literal "MOCK RAG ... in production this would call Materials Project API".
- **Guardrail pedagógico desconectado**: `socratic_debugger.py` solo está conectado a 1 de las 6 unidades.
- **Sin autoevaluación ejecutable, sin Diccionario de Variables sistemático, sin política de IA para el alumno, sin bibliografía citada (0 DOIs/arXiv en todo el corpus)**.
- **Solo 3 archivos de test**, frente a 208 (LP) y 167 (Probabilidad).

Además, en esta sesión se identificó un problema de **cohesión temática**: la Unidad 5 (Sistemas Multi-Agente, ~1.65MB de notebooks — LangChain/LangGraph, CrewAI, Google ADK/A2A, RAG/GraphRAG, multimodal) es contenido genérico de ingeniería de agentes de IA sin ninguna conexión con nanotecnología. Las Unidades 3 (ML aplicado) y 4 (IA aplicada a datos experimentales) están en un punto intermedio: mezclan fundamentos técnicos genéricos (ML clásico, redes neuronales, optimización bayesiana) con ejemplos ya aplicados a nanotecnología (dataset de nanopartículas, predicción de bandgap, detección de partículas por SEM, espectroscopía, óxidos de Materials Project).

## Decisión de alcance

Esta spec cubre **dos repos**, no uno:

1. **Un repo nuevo**, `AI-Agentic-Systems-Core` (bajo la organización GitHub `Multiagent-AI-Lab`, igual que los otros 3), que se convierte en un curso completo y genérico de IA/ML/Sistemas Multi-Agente — sin nanotecnología — con la misma calidad estructural que LP y Probabilidad.
2. **Antigravity-Nano-Research-Multiagentic-Core, nivelado**, que conserva su identidad de curso de nanotecnología pura, adopta la infraestructura de agentes real ya construida y probada en el repo nuevo, y reescribe su contenido para depender explícitamente del repo nuevo como prerequisito curricular (mismo patrón de "Prerequisitos de esta unidad" que ya usan LP y Probabilidad entre sus propias unidades).

## Mapeo de contenido educativo

| Unidad actual de Antigravity-Nano | Destino | Tratamiento |
|---|---|---|
| — (nueva, no existía) | **Solo `AI-Agentic-Systems-Core`, como Unidad 0** | Ver "Fuente adicional: Fundamentos Matemáticos de la IA (EMALCA)" abajo |
| U1 (Modelado a Nanoescala) | Antigravity-Nano | Se queda tal cual — ya es nano puro |
| U2 (Simulación Molecular / DFT) | Antigravity-Nano | Se queda tal cual — ya es nano puro |
| U3 (ML aplicado a nanomateriales) | **Ambos repos** | Los fundamentos de ML clásico/redes neuronales se generalizan y viven en `AI-Agentic-Systems-Core` (sin datos de nanopartículas); Antigravity-Nano conserva y profundiza sus ejemplos aplicados (dataset de nanopartículas, predicción de bandgap), citando el repo nuevo como prerequisito de la teoría |
| U4 (IA aplicada a datos experimentales) | **Ambos repos** | Mismo tratamiento que U3: fundamentos de optimización bayesiana/detección de anomalías generalizados al repo nuevo; Antigravity-Nano conserva SEM/espectroscopía/óxidos, citando el repo nuevo |
| U5 (Sistemas Multi-Agente) | **Solo `AI-Agentic-Systems-Core`** | Se traslada completa, tal cual (ya es genérica) — LangChain/LangGraph, CrewAI, Google ADK/A2A, RAG/GraphRAG, multimodal. Se elimina de Antigravity-Nano. Se funde (sin duplicar) con los Capítulos 9-10bis de la fuente EMALCA (ver abajo), que cubren el mismo dominio desde el ángulo matemático formal |
| U6 (Proyecto Integrador) | Antigravity-Nano | Se queda, reescrito para integrar explícitamente herramientas del repo nuevo (agentes, RAG) aplicadas a un problema real de nanotecnología |

## Fuente adicional: Fundamentos Matemáticos de la IA (EMALCA)

En `C:\Users\ljyud\Desktop\Fundamentos Matemáticos de la Inteligencia Artificial\` existe material propio del profesor (impartido en EMALCA Ecuador 2026, Yachay Tech) con rigor matemático formal: `notas_fundamentos_matematicos_ia.md` (1,505 líneas), 11 capítulos (0: El Mapa; 1: Álgebra Lineal; 2: Cálculo Multivariado/Backprop; 3: Sistemas Dinámicos/Neural ODEs; 4: Probabilidad/Procesos Estocásticos; 5: Teoría de la Información/XAI; 6: Geometría del Transformer; 7: Geometría Diferencial/Transporte Óptimo; 8: Topología Algebraica; 9: Sistemas Multiagente/Teoría de Juegos; 10: Lógica/Grafos; 10-bis: Ingeniería de Agentes), más gráficos generados (SVD, cálculo, dinámica, transformer, grafo bipartito, DAG causal) y un PDF derivado.

Este material se incorpora a `AI-Agentic-Systems-Core` de dos formas simultáneas, no excluyentes:

1. **Como Unidad 0 curricular** ("Fundamentos Matemáticos"), adaptada al mismo estándar del resto del curso (patrón pedagógico central, contexto aplicado, Diccionario de Variables, autoevaluación) — precede a las unidades de ML/Redes Neuronales/Sistemas Multi-Agente, dándoles una base formal que hoy ninguno de los 3 repos hermanos tiene con este nivel de rigor. Los Capítulos 9 y 10-bis (Sistemas Multiagente, Ingeniería de Agentes) se funden con el contenido migrado de U5 de Antigravity-Nano en vez de duplicarse — un solo tratamiento del tema, no dos.
2. **Como fuente indexada en el RAG del `TutorAgent`** del repo nuevo, junto con el contenido propio de las unidades y los PDFs de bibliografía académica ya planeados — el mismo patrón que LP/Probabilidad ya usan para sus propias lecciones.

El `GOVERNANCE.md` que acompaña hoy esa carpeta de EMALCA es una copia idéntica del `GOVERNANCE.md` original (pre-corrección) de Antigravity-Nano — mismo "Consejo de 7", mismo texto de "Nanotecnología"/`ia_nano` fuera de lugar para contenido de matemáticas puras. No se reutiliza; confirma que esa plantilla de gobernanza ficticia se copió entre proyectos del profesor más allá de Antigravity-Nano, reforzando por qué el `GOVERNANCE.md` honesto de esta spec (ver más abajo) importa para ambos repos nuevos/nivelados.

## Mapeo del "Consejo de 7" a agentes reales

Ningún rol se descarta sin antes evaluar qué utilidad real tiene detrás y si es rescatable:

| Rol del Consejo (hoy: prosa/skill sin código propio) | Utilidad real que usa hoy | Absorbido por (agente nuevo) |
|---|---|---|
| @Architect | Ninguna (rol de coordinación en prosa) | `OrchestratorAgent` |
| @Scientist | Ninguna | `ContentAuditorAgent` |
| @Engineer | Ninguna | `CodeAuditorAgent` + `NotebookCompilerAgent` |
| @Safety_Gate | `stability_guardian.py` (real, 51 líneas, rescatable tal cual), `toxicity_predictor.py` (mock explícito de DeepChem), `socratic_debugger.py` (conectado a 1/6 unidades) | `SafetyGateAgent` (nuevo) — envuelve `stability_guardian` rescatado, reemplaza el mock de toxicidad con reglas estructurales reales vía RDKit, reconecta `socratic_debugger` a todas las unidades |
| @Analyst | Ninguna (`descriptor_miner`/`data-storytelling` no existen como código) | `EvaluatorAgent` |
| @Librarian | `librarian_rag.py` (mock de 4 compuestos) | `TutorAgent` — RAG real (ChromaDB + Gemini) + función real de consulta a Materials Project API (gratuita, sin key) |
| @QA | Ninguna (`systematic-debugging`/`code-review-excellence` no existen como código) | `OrchestratorAgent` (paso final de reporte) |

**Utilidades reales de `external_skills/` que se rescatan tal cual** (siguen viviendo ahí, importadas por los agentes nuevos, sin reescritura): `agent_warmup/context_loader.py`, `apis/github_skill_loader.py`, `apis/token_budget_guard.py`, `evaluation/output_scorer.py`, `memory/episodic_retriever.py`, `memory/graph_memory.py`, `numerical/basis_set_architect.py`, `numerical/stability_guardian.py`, `observability/trace_annotator.py`, `routing/task_classifier.py`.

**Mocks que se reemplazan por implementación real**: `ai_mining/toxicity_predictor.py`, `orchestration/librarian_rag.py`.

## Agentes a construir (idénticos en ambos repos, construidos primero en `AI-Agentic-Systems-Core`)

Mismo set que LP/Probabilidad, en `src/multiagent_core/`:

1. `NotebookCompilerAgent` — compila `.md` → `.ipynb`, estableciendo el `.md` como fuente de verdad (hoy el contenido vive directo en notebooks).
2. `ContentAuditorAgent` — audita teoría completa, LaTeX, contexto de dominio, Diccionario de Variables.
3. `CodeAuditorAgent` — análisis estático PEP8/OWASP.
4. `FlowchartAgent` + `PseudocodeAgent` — para el patrón central del repo nuevo (Hilo de Oro o equivalente propio).
5. `SafetyGateAgent` — nuevo, propio de este dominio (nanotecnología/ciencia), ver mapeo arriba.
6. `EvaluatorAgent` — calificación contra rúbrica.
7. `OrchestratorAgent` — coordina todos los anteriores en un reporte único.
8. `TutorAgent` — RAG real (ChromaDB + Gemini) sobre el contenido propio del repo **y** una carpeta de PDFs de bibliografía académica (cerrando "0 DOIs" del diagnóstico), más integración real con Materials Project API.

## Entorno de ejecución

**Revisado tras el Sub-proyecto 1**: cada repo usa su propio entorno conda, no uno compartido. `AI-Agentic-Systems-Core` usa `sys_agents` (Python 3.11, creado en el Sub-proyecto 1); Antigravity-Nano conserva su `ia_nano` propio (`environment.yml` ya declarado ahí). Decisión explícita: aunque ambos parten de una base similar, cada repo acumulará dependencias distintas conforme avancen sus sub-proyectos de contenido — genéricas de IA/ML/Sistemas Multi-Agente en uno (`langchain`, `crewai`, `google-adk`, `chromadb`, `sentence-transformers`), específicas de nanotecnología en el otro (`ase`, `rdkit`, `openmm`, `mp_api`) — y compartir un solo entorno los habría acoplado innecesariamente además de mezclar ambos conjuntos de deps pesadas en una sola instalación. `torch` y `chromadb`/`sentence-transformers` se agregan a `sys_agents` cuando el Sub-proyecto 2/3 los necesite de verdad, no de antemano.

## Estándar de calidad (idéntico al ya aplicado en LP/Probabilidad)

- TDD estricto para todo agente nuevo — sin excepción, incluyendo los que envuelven utilidades ya rescatadas (el wrapper en sí se construye con TDD, aunque la lógica interna rescatada no se reescriba).
- `ruff` limpio, sin excepciones.
- Patrón pedagógico central nombrado y omnipresente (como el Hilo de Oro de LP / Ciclo de Verificación Triple de Probabilidad) — a definir su nombre propio para `AI-Agentic-Systems-Core` durante el brainstorming de su sub-proyecto de contenido.
- Diccionario de Variables por unidad, verificado contra código realmente ejecutado (mismo criterio estricto que causó los rechazos de revisión en Probabilidad esta sesión — nunca un símbolo listado que no se use en un ejemplo real).
- Autoevaluación ejecutable (pytest/ipytest) en cada unidad.
- Política de uso de IA documentada para el alumno (a decidir su enfoque: progresiva como LP, o permitida con verificación crítica como Probabilidad).
- `GOVERNANCE.md` honesto en ambos repos: documenta el patrón pedagógico real y el estándar de calidad, nunca un pipeline de agentes de generación que no corre — mismo principio que ya se aplicó corrigiendo el `GOVERNANCE.md` de Probabilidad en esta sesión.
- Contexto de dominio obligatorio: nanotecnología en Antigravity-Nano; en `AI-Agentic-Systems-Core`, ejemplos genéricos de ingeniería de software/datos son aceptables (es un curso de fundamentos, no de un dominio de aplicación específico), pero deben ser concretos y no triviales.

## Secuencia de sub-proyectos (cada uno con su propio spec/plan de implementación cuando le toque)

1. **Crear el repo `AI-Agentic-Systems-Core`** — estructura base, licencia, README inicial, CI, entorno `sys_agents` propio (completado — ver nota en "Entorno de ejecución").
2. **Infraestructura de agentes en `AI-Agentic-Systems-Core`** — los 8 agentes listados arriba, con TDD estricto, en `src/multiagent_core/`.
3. **Contenido pedagógico de `AI-Agentic-Systems-Core`** — Unidad 0: Fundamentos Matemáticos (adaptada de la fuente EMALCA), ML fundamentals (de U3 generalizado), IA aplicada genérica (de U4 generalizado), Sistemas Multi-Agente completo (U5 tal cual, fusionada con Capítulos 9/10-bis de EMALCA sin duplicar), con Hilo de Oro/patrón propio, Diccionario de Variables, autoevaluación, GOVERNANCE.md honesto. La fuente EMALCA se indexa también en el RAG del `TutorAgent` de este repo.
4. **Migrar/adaptar los agentes a Antigravity-Nano** — mismo patrón de replicación ya usado entre LP y Probabilidad (código adaptado, no un paquete compartido — decisión explícita de esta spec, para no introducir una dependencia de publicación de paquete entre repos).
5. **Gobernanza honesta en Antigravity-Nano** — `GOVERNANCE.md` reescrito, reflejando los agentes reales ya migrados (sin el Consejo de 7 ficticio).
6. **Contenido pedagógico de Antigravity-Nano nivelado** — U1/U2 conservadas, U3/U4 reescritas citando `AI-Agentic-Systems-Core` como prerequisito y profundizando sus ejemplos aplicados, U5 eliminada de aquí, U6 reescrita integrando explícitamente las herramientas del repo nuevo.

Cada sub-proyecto pasa por su propio ciclo: brainstorming (si su alcance no está ya completamente fijado por esta spec) → `writing-plans` → `subagent-driven-development`, igual que las rondas ya cerradas de LP y Probabilidad en esta sesión.

## Activos existentes que la spec original pasó por alto (agregado tras revisión)

Una relectura del repo real encontró 4 elementos que ya existen en Antigravity-Nano y que el mapeo de contenido de arriba no cubre — cada uno necesita una decisión explícita antes de tocar U3/U4/U5/U6 en el Sub-proyecto 6, para no dejarlos huérfanos o desactualizados:

1. **Paper académico en `paper/`** (`paper.md`, `paper.bib`, 2 figuras) — formato JOSS/JOSE, con ORCID del profesor, fecha 2026, título "*Antigravity Nano Research Multiagentic Core: An Open Educational Framework for Multi-Agent AI in Computational Nanotechnology*". El paper describe la arquitectura multi-agente y RAG que este plan construye, pero que no existía de verdad en el código hasta ahora. **Confirmado con el usuario: nunca fue sometido, sigue siendo borrador local** — sin problema de integridad académica que resolver. Se actualiza como última tarea del Sub-proyecto 6 (una vez que la arquitectura real ya esté construida y el paper pueda describirla con precisión) y se somete solo entonces, nunca antes.
2. **Documentación multiidioma con MkDocs Material** (`docs/es`, `docs/de`, `docs/fr`, `docs/hi`, `docs/ja`, `docs/pt`, `docs/zh`, publicada en GitHub Pages vía `docs.yml`) — un activo que **ni LP ni Probabilidad tienen**, y que sería un retroceso perder. `docs/units/unit1.md`...`unit6.md` son espejos del contenido de las unidades para ese sitio — si U5 se elimina y U3/U4 se reescriben en el Sub-proyecto 6, estos espejos (y sus traducciones) quedan desactualizados o rotos si no se actualizan en el mismo sub-proyecto. Añadir explícitamente a las tareas del Sub-proyecto 6: actualizar `docs/units/*.md` (al menos la versión en español; las traducciones a los otros 6 idiomas quedan fuera de alcance salvo que el usuario pida lo contrario, documentándolo como deuda conocida en vez de dejarlo un olvido silencioso).
3. **Notebook de uso del sistema multiagente** (`notebooks_extra/USO_SISTEMA_MULTIAGENTE.ipynb`) — LP y Probabilidad SÍ tienen este entregable; Antigravity-Nano no. Agregarlo explícitamente como entregable del Sub-proyecto 4 (migración de agentes), no dejarlo implícito.
4. **Empaquetado (`pyproject.toml`) solo declara `external_skills*`** (`[tool.setuptools.packages.find] include = ["external_skills*"]`) — un `src/multiagent_core/` nuevo (Sub-proyecto 2) NO se incluiría automáticamente en el paquete instalable. El CI actual corre `pip install -e ".[dev]"` seguido de `pytest tests/`; sin actualizar esta declaración, el CI fallaría con `ImportError` en cualquier test que importe un agente nuevo. Añadir explícitamente como primer paso técnico del Sub-proyecto 2: actualizar `[tool.setuptools.packages.find]` para incluir `src*` o `multiagent_core*`, verificando `pip install -e ".[dev]"` en local antes de commitear.

## Fuera de alcance de esta spec

- No se decide todavía el nombre del patrón pedagógico central de `AI-Agentic-Systems-Core` (se decide en el brainstorming del Sub-proyecto 3).
- No se decide todavía el enfoque exacto de la política de IA para el alumno en ninguno de los dos repos (se decide en los sub-proyectos de contenido correspondientes).
- No se construye un paquete Python compartido entre los 4 repos — decisión explícita de replicar código, no compartir una dependencia, consistente con cómo ya conviven LP y Probabilidad.
- No se toca LP ni Probabilidad en este trabajo.
