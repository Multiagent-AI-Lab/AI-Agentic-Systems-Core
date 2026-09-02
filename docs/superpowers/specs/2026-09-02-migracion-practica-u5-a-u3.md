# Migración de la práctica de U5 (Antigravity-Nano) a U3 (AI-Agentic-Systems-Core)

## Contexto

`AI-Agentic-Systems-Core` es el repo base compartido de sistemas multi-agente,
del que depende `Antigravity-Nano-Research-Multiagentic-Core` (ambos cursos se
imparten simultáneamente en la UCEMICH, con audiencias distintas: alguien
puede cursar solo el repo de agentes, nadie cursa solo nanotecnología sin él).

La Unidad 3 de este repo (`lecciones/UNIDAD_3_SISTEMAS_MULTI_AGENTE.md`) cubre
Sistemas Multi-Agente a nivel conceptual y comparativo: selección de
arquitectura (frameworks, protocolos MCP/A2A, SDKs de proveedor), diseño
(Context Engineering, Spec-Driven Dev), evaluación (causalidad), despliegue
(observabilidad), y seguridad (3 vectores de ataque 2026). No tiene
profundidad práctica de implementación en los frameworks que menciona.

La Unidad 5 de Antigravity-Nano (`educational_content/unit_05_multi_agent_sys/`)
tiene 8 notebooks (más 1 meta-notebook) con implementación extensa y
validada (marzo 2026) de exactamente esos frameworks: LangGraph con
checkpoints reales, CrewAI con roles reales, Google ADK/A2A, RAG con ChromaDB,
GraphRAG con Neo4j, multimodal con FastAPI async, y un proyecto integrador.
Su propio README declara que el dominio de nanotecnología es "ilustrativo" —
la aplicación real a nanotecnología vive en su Unidad 6, no en la 5.

Hay solapamiento temático real entre ambas unidades sin que ninguna sea
completa por sí sola: U3 tiene la teoría sin práctica, U5 tiene la práctica
sin el marco conceptual que la conecta a Selección de Arquitectura, Evaluación
(causalidad) y Seguridad ya escritos en U3.

**Hallazgo de auditoría (revisando imports reales de los 8 notebooks antes
de escribir el plan):** los notebooks no son autónomos. Dependen de 7
módulos de `external_skills/` de Antigravity-Nano — infraestructura de
agentes genérica (no específica de nanotecnología), cargada casi siempre vía
`sys.path.insert` relativo a la posición del notebook en ese árbol:

| Módulo | Usado por |
|---|---|
| `external_skills.agent_warmup.context_loader` | U5_01, U5_02, U5_03 |
| `external_skills.routing.task_classifier` | U5_02, U5_03, U5_07 |
| `external_skills.evaluation.output_scorer` | U5_03, U5_08 |
| `external_skills.memory.episodic_retriever` | U5_05, U5_08 |
| `external_skills.observability.trace_annotator` | U5_02, U5_08 |
| `external_skills.apis.token_budget_guard` | U5_04, U5_05, U5_08 |
| `external_skills.apis.github_skill_loader` | U5_07 |

`external_skills/registry.py` (263 líneas) es el cargador versionado que
referencia estos módulos por nombre (`load_skill("episodic_retriever")`,
etc.) — también migra, pero **filtrado**: conserva solo las 7 entradas de
la tabla de arriba. Sus otras dos entradas (`graph_memory`,
`github_skill_loader`→`orchestration`/`librarian_rag`) no son importadas por
ningún notebook de los 8 migrados y se quedan en Antigravity-Nano.

`U5_04_GOOGLE_ADK_A2A_COMP.ipynb` además tiene una ruta absoluta
hardcodeada de una máquina distinta
(`d:/Users/UCEMICH/Desktop/antigravity projects/...`) en un
`load_dotenv()` — debe corregirse a una ruta relativa/`Path` durante la
migración, no preservarse tal cual.

**Segundo hallazgo:** `episodic_retriever.py` fija `_LOCAL_STORE_PATH` como
constante de módulo evaluada al importar
(`Path(os.environ.get("EPISODIC_STORE_PATH", Path.home() / ...))`) — viola
el patrón ya establecido en este repo (`TutorAgent`: `chroma_path`/
`memory_path` inyectables en el constructor, ver CLAUDE.md) y hace
imposible aislar tests con `tmp_path`. Además `add_episode` tiene un
`except Exception: pass` genérico al llamar al cliente Mem0 (silent
failure). Ambos se corrigen durante la migración: la ruta de
almacenamiento pasa a ser un parámetro con default, y la excepción se
acota a las que el SDK de Mem0 puede lanzar razonablemente
(`ConnectionError`, o la excepción específica del SDK si Mem0 expone una),
registrando el fallback con `logging` en vez de silenciarlo.

## Objetivo

Unificar ambas en una sola U3 completa dentro de `AI-Agentic-Systems-Core`:
teoría (ya existente) + práctica (migrada desde U5). Sin duplicar contenido,
sin reescribir el `.md` teórico ya publicado, sin tocar aún la U5 de origen.

## Alcance

**Incluye:**
- Copiar 8 notebooks de Antigravity-Nano a `AI-Agentic-Systems-Core`, con
  numeración `U3_` en vez de `U5_`.
- Migrar los 7 módulos de `external_skills/` de los que dependen (tabla en
  Contexto) a `src/multiagent_core/skills/`, junto con `registry.py`
  filtrado a esas 7 entradas.
- Corregir la ruta absoluta hardcodeada en `U5_04` a una ruta relativa al
  propio repo.
- Agregar un extra `pyproject.toml` (`practica-u3`) con las versiones exactas
  con las que esos notebooks fueron validados — no unificar con las
  versiones ya fijadas para el resto de `sys_agents` (más nuevas).
- Actualizar `lecciones/UNIDAD_3_SISTEMAS_MULTI_AGENTE.md` con una sección
  nueva que referencie los notebooks prácticos migrados, conectándolos
  explícitamente con las fases del Ciclo del Agente ya escritas ahí.
- Actualizar `README.md` del repo con la nueva estructura de U3 (teoría +
  práctica).
- Tests: verificar que los notebooks migrados son JSON válido y que
  `ContentAuditorAgent`/`NotebookCompilerAgent` no los rompen ni intentan
  regenerarlos desde `.md` (son notebooks de autor, no derivados).

**Explícitamente fuera de alcance (este cambio):**
- Eliminar U5 de Antigravity-Nano — **NO se toca ni se elimina nada en el
  árbol de Antigravity-Nano en este trabajo.** Queda para una decisión
  posterior y explícita del usuario.
- `SafetyGateAgent` con especialización química (RDKit) — pertenece al
  Sub-proyecto 4 original (agentes core), no a esta migración de contenido.
- Reescribir el dominio de ejemplos de los notebooks migrados — se preservan
  tal cual, dado que ya son "ilustrativos"/genéricos por diseño.
- `U5_00_META_CONSTRUYENDO_CON_IA.ipynb` — es meta-contenido sobre el proceso
  de autoría de Antigravity-Nano, no migra.
- Duplicados de variante de modelo (`U5_04_GOOGLE_ADK_A2A_COMP` vs
  `U5_04_GOOGLE_ADK_A2A_COMP_GEMMA4`): migra solo la versión canónica
  `U5_04_GOOGLE_ADK_A2A_COMP.ipynb`.

## Diseño

### Archivos migrados (origen → destino)

| Origen (Antigravity-Nano) | Destino (AI-Agentic-Systems-Core) |
|---|---|
| `U5_01_FUNDAMENTOS_AGENTES_MODERNOS.ipynb` | `notebooks/practica_u3/U3_01_FUNDAMENTOS_AGENTES_MODERNOS.ipynb` |
| `U5_02_LANGCHAIN_AVANZADO_LANGGRAPH.ipynb` | `notebooks/practica_u3/U3_02_LANGCHAIN_AVANZADO_LANGGRAPH.ipynb` |
| `U5_03_CREWAI_SISTEMAS_MULTIAGENTE.ipynb` | `notebooks/practica_u3/U3_03_CREWAI_SISTEMAS_MULTIAGENTE.ipynb` |
| `U5_04_GOOGLE_ADK_A2A_COMP.ipynb` | `notebooks/practica_u3/U3_04_GOOGLE_ADK_A2A_COMP.ipynb` |
| `U5_05_RAG_MEMORIA_AVANZADA.ipynb` | `notebooks/practica_u3/U3_05_RAG_MEMORIA_AVANZADA.ipynb` |
| `U5_06_GRAPH_RAG_MEMORIA.ipynb` | `notebooks/practica_u3/U3_06_GRAPH_RAG_MEMORIA.ipynb` |
| `U5_07_MULTIMODAL_PRODUCCION.ipynb` | `notebooks/practica_u3/U3_07_MULTIMODAL_PRODUCCION.ipynb` |
| `U5_08_PROYECTO_INTEGRADOR.ipynb` | `notebooks/practica_u3/U3_08_PROYECTO_INTEGRADOR.ipynb` |
| `requirements.txt` | referencia para armar `practica-u3` en `pyproject.toml` (no se copia tal cual: se reconcilia con las dependencias base ya presentes) |

`notebooks/practica_u3/` es un subdirectorio nuevo, separado de
`notebooks/*.ipynb` (los 5 generados automáticamente desde `lecciones/`) —
marca visualmente que estos NO se regeneran desde Markdown.

### Módulos de `external_skills/` migrados

| Origen (Antigravity-Nano) | Destino (AI-Agentic-Systems-Core) |
|---|---|
| `external_skills/agent_warmup/context_loader.py` | `src/multiagent_core/skills/agent_warmup/context_loader.py` |
| `external_skills/routing/task_classifier.py` | `src/multiagent_core/skills/routing/task_classifier.py` |
| `external_skills/evaluation/output_scorer.py` | `src/multiagent_core/skills/evaluation/output_scorer.py` |
| `external_skills/memory/episodic_retriever.py` | `src/multiagent_core/skills/memory/episodic_retriever.py` |
| `external_skills/observability/trace_annotator.py` | `src/multiagent_core/skills/observability/trace_annotator.py` |
| `external_skills/apis/token_budget_guard.py` | `src/multiagent_core/skills/apis/token_budget_guard.py` |
| `external_skills/apis/github_skill_loader.py` | `src/multiagent_core/skills/apis/github_skill_loader.py` |
| `external_skills/registry.py` (filtrado a 7 entradas) | `src/multiagent_core/skills/registry.py` |

Cada notebook migrado cambia su `sys.path.insert`/import relativo por un
import absoluto de paquete: `from src.multiagent_core.skills.<paquete>.<modulo> import ...`,
consistente con la convención de imports del resto de este repo
(`from src.multiagent_core.tutor_agent import TutorAgent`, etc. — ver
CLAUDE.md). Los tests de cada módulo migrado (clases correspondientes de
`tests/test_external_skills.py` de origen: `TestContextLoader`,
`TestTaskClassifier`, `TestOutputScorer`, `TestEpisodicRetriever`,
`TestTraceAnnotator`, `TestTokenBudgetGuard`, y las 4 pruebas de
`TestRegistry` — ninguna de las cuales ejercita `graph_memory` ni
`github_skill_loader`, así que migran sin ajuste de alcance) migran a
`tests/skills/test_<modulo>.py` en este repo, adaptando solo la ruta de
import. `github_skill_loader.py` no tiene clase de test dedicada en el
origen (deuda preexistente, no introducida por esta migración) — migra sin
tests nuevos, documentado como deuda heredada en el reporte de la tarea que
lo mueva.

### `pyproject.toml` — extra `practica-u3`

Nuevo grupo `[project.optional-dependencies]` con las versiones exactas de
`requirements.txt` de origen, filtrando lo que el curso de práctica de U3
efectivamente usa (excluye SDKs comparativos sin ejercicio: `pyautogen`,
`openai-agents`, proveedores no usados como `dashscope`/`xai-sdk`/`pinecone-client`
si los notebooks no los importan — se confirma en Tarea 1 revisando los
imports reales de los 8 notebooks antes de fijar la lista final).

Instalación: `pip install -e ".[practica-u3]"`, adicional a la instalación
base — quien solo curse la parte teórica de U3 no necesita este peso.

### Sección nueva en `UNIDAD_3_SISTEMAS_MULTI_AGENTE.md`

Se agrega, después de la sección de Seguridad y antes del Diccionario de
Variables, una sección `## Notebooks de Práctica` con:
- Tabla de los 8 notebooks migrados, cada uno anotado con qué fase del Ciclo
  del Agente profundiza (ej. `U3_02` → Implementación con LangGraph;
  `U3_05`/`U3_06` → Implementación, memoria persistente).
- Instrucción de instalación (`pip install -e ".[practica-u3]"`).
- Nota explícita: estos notebooks son de autoría directa (no derivados de
  `lecciones/*.md`) y no se regeneran con `convert_to_notebooks.py`.

### Tests

- Test que verifica que los 8 archivos existen en `notebooks/practica_u3/`
  y son JSON/notebook válido (`nbformat.read` sin excepción).
- Test que verifica que `convert_to_notebooks.py`/`NotebookCompilerAgent` no
  itera `notebooks/practica_u3/` (evitar que un futuro cambio intente tratar
  estos notebooks de autor como si fueran generados).
- Test que verifica que ningún notebook migrado en `notebooks/practica_u3/`
  contiene una ruta absoluta de sistema de archivos en su fuente (regex
  sobre `[A-Za-z]:[\\/]` o `/home/`/`/Users/` fuera de comentarios de
  ejemplo) — cubre la regresión de la ruta hardcodeada de `U5_04`.
- Tests de los 7 módulos migrados de `external_skills/` (ver tabla arriba)
  pasan en `tests/skills/` importando desde
  `src.multiagent_core.skills.*`.

## Global Constraints

- No se modifica ni se elimina nada en el árbol de trabajo de
  `Antigravity-Nano-Research-Multiagentic-Core` en esta spec.
- No se actualizan las versiones de los paquetes migrados a las versiones
  más nuevas ya fijadas para el resto de `sys_agents` — se preservan las de
  validación original (marzo 2026) para no arriesgar romper notebooks ya
  probados.
- El dominio de ejemplos de los notebooks migrados no se reescribe.
- `U5_00` (meta-notebook) y la variante `_GEMMA4` de `U5_04` no migran.
- `external_skills/graph_memory.py` y la entrada `github_skill_loader`→`orchestration`/`librarian_rag`
  de `registry.py` NO migran — ningún notebook de los 8 los importa.
- La ruta absoluta hardcodeada en `U5_04` se corrige durante la migración
  (no se preserva como estaba en origen).
