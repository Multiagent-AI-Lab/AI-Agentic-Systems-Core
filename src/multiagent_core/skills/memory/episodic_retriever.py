"""src/multiagent_core/skills/memory/episodic_retriever.py
Memoria episódica para agentes: Mem0 (cloud) con fallback a JSON local.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SKILL_METADATA = {
    "name": "episodic_retriever",
    "domain": "memory",
    "description": "Memoria episódica para agentes. Backend: Mem0 (cloud) o JSON local (fallback).",
    "version": "1.0.0",
    "input": "content: str, user_id: str, metadata: dict | None, store_path: Path | None",
    "output": "list[dict] con content, metadata, score",
    "dependencies": [],
}


def _default_store_path() -> Path:
    """Ruta de almacenamiento local por defecto (fallback), resuelta en
    tiempo de llamada — nunca como constante de módulo, para que los tests
    puedan inyectar tmp_path sin depender del orden de import."""
    return Path(
        os.environ.get(
            "EPISODIC_STORE_PATH",
            str(Path.home() / ".cache" / "ia_nano" / "episodic_memory.json"),
        )
    )


# Cache del cliente Mem0 (lazy init)
_mem0_client = None
_mem0_available = False


def _get_mem0_client():
    """Intenta inicializar el cliente Mem0 una sola vez."""
    global _mem0_client, _mem0_available
    if _mem0_client is not None:
        return _mem0_client

    try:
        from mem0 import MemoryClient

        api_key = os.environ.get("MEM0_API_KEY")
        if api_key:
            _mem0_client = MemoryClient(api_key=api_key)
            _mem0_available = True
        else:
            _mem0_available = False
    except ImportError:
        _mem0_available = False

    return _mem0_client


def _load_local_store(store_path: Path) -> dict[str, list[dict]]:
    """Carga el almacén local de JSON."""
    if store_path.exists():
        try:
            return json.loads(store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_local_store(store: dict[str, list[dict]], store_path: Path) -> None:
    """Guarda el almacén local de JSON."""
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_episode(
    content: str,
    user_id: str,
    metadata: dict[str, Any] | None = None,
    store_path: Path | None = None,
) -> None:
    """
    Agrega un episodio a la memoria del usuario.

    Args:
        content: Texto del episodio a recordar.
        user_id: Identificador del usuario/agente.
        metadata: Metadatos opcionales (topic, type, timestamp, etc).
        store_path: Ruta del almacén JSON local. Si es None, usa
            `_default_store_path()`. Inyectable para aislar tests con
            tmp_path.
    """
    resolved_path = store_path if store_path is not None else _default_store_path()

    client = _get_mem0_client()
    if _mem0_available and client:
        try:
            client.add(content, user_id=user_id, metadata=metadata or {})
            return
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning("Mem0 add() fallo, usando fallback JSON local: %s", e)

    # Fallback: almacén JSON local
    store = _load_local_store(resolved_path)
    user_episodes = store.setdefault(user_id, [])
    user_episodes.append(
        {
            "id": str(uuid.uuid4()),
            "content": content,
            "metadata": metadata or {},
        }
    )
    # Mantener solo los últimos 100 episodios por usuario
    store[user_id] = user_episodes[-100:]
    _save_local_store(store, resolved_path)


def retrieve(
    query: str,
    user_id: str,
    top_k: int = 3,
    store_path: Path | None = None,
) -> list[dict]:
    """
    Recupera los episodios más relevantes para un query.

    Args:
        query: Query de búsqueda.
        user_id: Identificador del usuario/agente.
        top_k: Número máximo de episodios a retornar.
        store_path: Ruta del almacén JSON local. Si es None, usa
            `_default_store_path()`. Inyectable para aislar tests con
            tmp_path.

    Returns:
        Lista de dicts con: content, metadata, score (0-1).
    """
    resolved_path = store_path if store_path is not None else _default_store_path()

    client = _get_mem0_client()
    if _mem0_available and client:
        try:
            results = client.search(query, user_id=user_id, limit=top_k)
            return [
                {
                    "content": r.get("memory", r.get("content", "")),
                    "metadata": r.get("metadata", {}),
                    "score": r.get("score", 0.5),
                }
                for r in results
            ]
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning("Mem0 search() fallo, usando fallback JSON local: %s", e)

    # Fallback: búsqueda lineal por palabras clave en JSON local
    store = _load_local_store(resolved_path)
    episodes = store.get(user_id, [])

    query_words = set(query.lower().split())
    scored = []
    for ep in episodes:
        ep_words = set(ep["content"].lower().split())
        overlap = len(query_words & ep_words)
        score = overlap / max(len(query_words), 1)
        if score > 0:
            scored.append(
                {
                    "content": ep["content"],
                    "metadata": ep.get("metadata", {}),
                    "score": round(score, 3),
                }
            )

    # Ordenar por score descendente y retornar top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_for_context(episodes: list[dict]) -> str:
    """
    Formatea episodios recuperados como contexto para un LLM.

    Args:
        episodes: Lista de episodios retornados por retrieve().

    Returns:
        String formateado listo para incluir en un prompt de sistema.
    """
    if not episodes:
        return ""

    lines = ["**Contexto de sesiones anteriores (memoria episódica):**"]
    for i, ep in enumerate(episodes, 1):
        content = ep.get("content", "")
        score = ep.get("score", 0.0)
        lines.append(f"  [{i}] (relevancia: {score:.2f}) {content}")

    return "\n".join(lines)
