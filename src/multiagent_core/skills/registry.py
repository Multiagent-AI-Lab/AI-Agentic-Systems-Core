"""
src/multiagent_core/skills/registry.py
=======================================
Registro central de skills reutilizables del proyecto IA Nanotecnología.

Soporta:
- Carga de skills por nombre: load_skill("episodic_retriever")
- Carga por versión: load_skill("episodic_retriever@1.0.0")
- Alias "latest" automático: load_skill("episodic_retriever") → carga la versión más reciente
- Descubrimiento de todas las skills disponibles: discover_skills()

Uso básico:
    from src.multiagent_core.skills.registry import load_skill, SKILL_REGISTRY

    retrieve = load_skill("episodic_retriever")
    episodes = retrieve.retrieve(query="nanoparticles", user_id="user_1")

Uso avanzado (versión específica):
    retrieve_v1 = load_skill("episodic_retriever@1.0.0")
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Registro canónico de skills
# ─────────────────────────────────────────────────────────────────────────────
# Formato: "skill_name@version" → {module, description, author, tags}
# La clave sin versión ("skill_name") es un alias a "skill_name@latest".
#
# Añadir una nueva skill:
#   1. Crear el archivo .py en el módulo correspondiente
#   2. Agregar la entrada aquí con versión, módulo, descripción y tags
#   3. Actualizar el alias "latest" si es la versión más reciente
#   4. Escribir tests en tests/skills/

SKILL_REGISTRY: dict[str, dict[str, Any]] = {
    # ── agent_warmup ────────────────────────────────────────────────────────
    "context_loader@1.0.0": {
        "module": "src.multiagent_core.skills.agent_warmup.context_loader",
        "description": "Inyecta contexto de dominio en agentes LLM. Soporta 5 dominios predefinidos y contexto custom.",
        "author": "IA Nanotecnología Team",
        "version": "1.0.0",
        "tags": ["warmup", "context", "agent"],
        "requires": [],
    },
    # ── routing ─────────────────────────────────────────────────────────────
    "task_classifier@1.0.0": {
        "module": "src.multiagent_core.skills.routing.task_classifier",
        "description": "Clasifica tareas en categorías usando reglas y LLM como fallback.",
        "author": "IA Nanotecnología Team",
        "version": "1.0.0",
        "tags": ["routing", "classification", "llm"],
        "requires": [],
    },
    # ── evaluation ──────────────────────────────────────────────────────────
    "output_scorer@1.0.0": {
        "module": "src.multiagent_core.skills.evaluation.output_scorer",
        "description": "Evalúa outputs de LLMs con juez LLM y scoring heurístico.",
        "author": "IA Nanotecnología Team",
        "version": "1.0.0",
        "tags": ["evaluation", "scoring", "llm-judge"],
        "requires": [],
    },
    # ── memory ───────────────────────────────────────────────────────────────
    "episodic_retriever@1.0.0": {
        "module": "src.multiagent_core.skills.memory.episodic_retriever",
        "description": "Memoria episódica para agentes. Backend: Mem0 (cloud) o JSON local (fallback).",
        "author": "IA Nanotecnología Team",
        "version": "1.0.0",
        "tags": ["memory", "episodic", "mem0"],
        "requires": [],
    },
    # ── observability ────────────────────────────────────────────────────────
    "trace_annotator@1.0.0": {
        "module": "src.multiagent_core.skills.observability.trace_annotator",
        "description": "Anotaciones de LangSmith y decorador @traced para funciones de agentes.",
        "author": "IA Nanotecnología Team",
        "version": "1.0.0",
        "tags": ["observability", "tracing", "langsmith"],
        "requires": ["langsmith"],
    },
    # ── apis ─────────────────────────────────────────────────────────────────
    "token_budget_guard@1.0.0": {
        "module": "src.multiagent_core.skills.apis.token_budget_guard",
        "description": "Control de presupuesto de tokens por sesión con circuit breaker.",
        "author": "IA Nanotecnología Team",
        "version": "1.0.0",
        "tags": ["budget", "tokens", "cost-control"],
        "requires": [],
    },
    "github_skill_loader@1.0.0": {
        "module": "src.multiagent_core.skills.apis.github_skill_loader",
        "description": "Carga skills desde GitHub con verificación de integridad SHA-256.",
        "author": "IA Nanotecnología Team",
        "version": "1.0.0",
        "tags": ["github", "skills", "loading", "security"],
        "requires": ["httpx"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Resolver alias "latest" automáticamente
# ─────────────────────────────────────────────────────────────────────────────
# Para cada skill_name@version, registrar también "skill_name" → última versión.
# Si hay varias versiones, gana la lexicográficamente mayor (ej. "1.1.0" > "1.0.0").


def _build_latest_aliases(registry: dict[str, dict]) -> dict[str, dict]:
    """Construye aliases skill_name → skill_name@latest."""
    latest: dict[str, tuple[str, dict]] = {}  # name → (version_str, entry)
    for key, entry in registry.items():
        if "@" not in key:
            continue  # ignorar alias ya cargados
        name, version = key.rsplit("@", 1)
        current_best = latest.get(name)
        if current_best is None or version > current_best[0]:
            latest[name] = (version, entry)

    aliases = {}
    for name, (version, entry) in latest.items():
        aliases[name] = {**entry, "_alias_of": f"{name}@{version}"}
    return aliases


# Poblar el registro con los alias latest
_LATEST_ALIASES = _build_latest_aliases(SKILL_REGISTRY)
SKILL_REGISTRY.update(_LATEST_ALIASES)


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────


def load_skill(skill_key: str) -> Any:
    """
    Carga e importa un módulo de skill desde el registro.

    Args:
        skill_key: Nombre de la skill con o sin versión.
                   Ejemplos: "episodic_retriever", "episodic_retriever@1.0.0"

    Returns:
        El módulo Python importado de la skill.

    Raises:
        KeyError: Si la skill no existe en el registro.
        ImportError: Si el módulo de la skill no puede importarse.
    """
    entry = SKILL_REGISTRY.get(skill_key)
    if entry is None:
        available = discover_skills()
        raise KeyError(
            f"Skill '{skill_key}' no encontrada en el registro. "
            f"Skills disponibles: {', '.join(available)}"
        )

    module_path = entry["module"]
    try:
        module = importlib.import_module(module_path)
        logger.debug("Skill cargada: %s → %s", skill_key, module_path)
        return module
    except ImportError as e:
        required = entry.get("requires", [])
        req_msg = f"Instalar con: pip install {' '.join(required)}" if required else ""
        raise ImportError(
            f"No se pudo importar la skill '{skill_key}' desde '{module_path}'. "
            f"{req_msg}\nError original: {e}"
        ) from e


def discover_skills(tag: str | None = None) -> list[str]:
    """
    Retorna los nombres de todas las skills registradas.

    Args:
        tag: Si se proporciona, filtra por tag (ej. "memory", "evaluation").

    Returns:
        Lista de claves de skills (incluyendo versiones y aliases latest).
    """
    keys = list(SKILL_REGISTRY.keys())
    if tag:
        keys = [k for k in keys if tag in SKILL_REGISTRY[k].get("tags", [])]
    return sorted(keys)


def describe_skill(skill_key: str) -> dict:
    """
    Retorna la metadata completa de una skill del registro.

    Args:
        skill_key: Nombre de la skill con o sin versión.

    Returns:
        Dict con module, description, version, tags, requires.
    """
    entry = SKILL_REGISTRY.get(skill_key)
    if entry is None:
        raise KeyError(f"Skill '{skill_key}' no encontrada en el registro.")
    return entry


def list_skills_by_tag() -> dict[str, list[str]]:
    """
    Agrupa las skills disponibles por tag.

    Returns:
        Dict tag → [lista de skill_keys].
    """
    result: dict[str, list[str]] = {}
    for key, entry in SKILL_REGISTRY.items():
        for tag in entry.get("tags", []):
            result.setdefault(tag, []).append(key)
    return {tag: sorted(keys) for tag, keys in sorted(result.items())}


if __name__ == "__main__":
    # Demostración del registro cuando se ejecuta directamente
    print("=== SKILL REGISTRY ===\n")
    print("Skills disponibles por tag:\n")
    by_tag = list_skills_by_tag()
    for tag, skills in by_tag.items():
        # Solo mostrar aliases (sin @version) para simplificar
        aliases = [s for s in skills if "@" not in s]
        if aliases:
            print(f"  [{tag}]")
            for skill in aliases:
                desc = SKILL_REGISTRY[skill].get("description", "")
                print(f"    - {skill}: {desc[:60]}...")

    print("\nTotal de skills registradas:", len([k for k in SKILL_REGISTRY if "@" not in k]))
