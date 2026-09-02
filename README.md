# AI-Agentic-Systems-Core

## Curso genérico de IA, Machine Learning y Sistemas Multi-Agente — UCEMICH

Repositorio hermano de [Programming-Logic-Agentic-AI-Development](https://github.com/Multiagent-AI-Lab/Programming-Logic-Agentic-AI-Development),
[Probability-Statistics-Agentic-AI-Core](https://github.com/Multiagent-AI-Lab/Probability-Statistics-Agentic-AI-Core)
y [Antigravity-Nano-Research-Multiagentic-Core](https://github.com/Multiagent-AI-Lab/Antigravity-Nano-Research-Multiagentic-Core).

Cubre los fundamentos genéricos de IA/ML/Sistemas Multi-Agente que antes
vivían mezclados con contenido de nanotecnología en Antigravity-Nano.
Antigravity-Nano depende de este repo como prerequisito curricular para
sus propias unidades aplicadas.

Ver `docs/superpowers/specs/` para el diseño completo de esta separación.

---

## 🚀 Configuración del Entorno de Trabajo

El proyecto usa su propio entorno conda `sys_agents`, con Python 3.11 —
independiente del `ia_nano` de Antigravity-Nano-Research-Multiagentic-Core.
Cada repo evoluciona sus propias dependencias (este repo es genérico de
IA/ML/Sistemas Multi-Agente; Antigravity-Nano es específico de
nanotecnología), así que no comparten entorno pese a ser repos hermanos.

### 1. Crear / Actualizar el Entorno Conda

```bash
conda env create -f environment.yml
```

o, si el entorno ya existe:

```bash
conda env update -f environment.yml
```

### 2. Activar el Entorno

```bash
conda activate sys_agents
```

### 3. Instalar el paquete en modo editable

```bash
pip install -e ".[dev]"
```

### 4. Correr la suite de tests

```bash
pytest tests/ -v --tb=short
```

## Estructura del curso

El contenido pedagógico en `lecciones/` cubre 5 unidades (U0-U4). U0-U3 son
práctica estandarizada (fundamentos y patrones ya consolidados en la
industria); U4 es Línea de Investigación (contenido de frontera, sin el
mismo nivel de estandarización de producción que U0-U3).

| Unidad | Título | Tipo |
|---|---|---|
| U0 | Fundamentos Matemáticos | Práctica estandarizada |
| U1 | ML Fundamentals | Práctica estandarizada |
| U2 | IA Aplicada Genérica | Práctica estandarizada |
| U3 | Sistemas Multi-Agente (teoría + práctica) | Práctica estandarizada |
| U4 | Sistemas Agénticos Adaptativos | **Línea de Investigación** |

La práctica de U3 vive en `notebooks/practica_u3/` (8 notebooks, ver
sección "Notebooks de Práctica" de `lecciones/UNIDAD_3_SISTEMAS_MULTI_AGENTE.md`)
e instala con `pip install -e ".[practica-u3]"`.

## Estructura del repositorio

```
src/multiagent_core/   # Agentes Python (ver docs/superpowers/specs/)
tests/                 # Suite de tests (pytest)
lecciones/             # Contenido pedagógico en Markdown (Sub-proyecto 3)
notebooks/             # Notebooks generados desde lecciones/ (no editar a mano)
docs/superpowers/      # Specs de diseño de este repo
```
