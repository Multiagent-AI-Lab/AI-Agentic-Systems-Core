"""src/multiagent_core/skills/observability/trace_annotator.py
Trazabilidad de agentes con LangSmith. Fallback a logging si no configurado.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

SKILL_METADATA = {
    "name": "trace_annotator",
    "domain": "observability",
    "description": (
        "Decorador para trazar llamadas de agentes con LangSmith. "
        "Fallback a console logging cuando LANGCHAIN_API_KEY no está configurado."
    ),
    "version": "1.0.0",
    "input": "project_name: str, notebook: str | None, concept: str | None",
    "output": "dict {status, message} o función decorada",
    "dependencies": [],
}

# Estado del cliente LangSmith
_langsmith_available = False
_current_project: str | None = None


def warm_up(project_name: str = "ia-nano-unit-05") -> dict[str, str]:
    """
    Inicializa la conexión con LangSmith.

    Args:
        project_name: Nombre del proyecto en LangSmith.

    Returns:
        dict con {status, message}.
    """
    global _langsmith_available, _current_project

    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        _langsmith_available = False
        _current_project = project_name
        logger.info(
            "LangSmith no configurado — tracing desactivado. Set LANGCHAIN_API_KEY para activar."
        )
        return {
            "status": "fallback",
            "message": (
                "LANGCHAIN_API_KEY no configurado. " "Trazabilidad redirigida a console logging."
            ),
        }

    try:
        from langsmith import Client  # type: ignore

        os.environ.setdefault("LANGCHAIN_PROJECT", project_name)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        client = Client()
        # Verificar conectividad básica
        _ = client.list_projects()
        _langsmith_available = True
        _current_project = project_name
        return {
            "status": "ok",
            "message": f"LangSmith listo. Proyecto: '{project_name}'.",
        }
    except ImportError:
        _langsmith_available = False
        _current_project = project_name
        return {
            "status": "fallback",
            "message": "langsmith no instalado. Instala con: pip install langsmith",
        }
    # Cualquier fallo de conectividad/configuración con LangSmith reportado
    # en el mensaje de retorno, nunca debe tumbar el warm_up.
    except Exception as exc:  # noqa: BLE001
        _langsmith_available = False
        _current_project = project_name
        return {
            "status": "warning",
            "message": f"LangSmith configurado pero error de conexión: {exc}",
        }


def traced(
    notebook: str | None = None,
    concept: str | None = None,
) -> Callable:
    """
    Decorador de trazabilidad para funciones de agentes.

    Args:
        notebook: Nombre del notebook origen (para metadatos).
        concept: Concepto educativo que ilustra esta traza.

    Returns:
        Decorador que envuelve la función con trazabilidad LangSmith
        o logging de consola como fallback.

    Example:
        @traced(notebook="U5_01", concept="tool_use")
        def my_agent_call(prompt):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run_name = f"{notebook or 'unknown'}/{func.__name__}"
            tags = []
            if notebook:
                tags.append(notebook)
            if concept:
                tags.append(concept)

            if _langsmith_available:
                try:
                    from langsmith import traceable  # type: ignore

                    traced_func = traceable(
                        run_type="chain",
                        name=run_name,
                        tags=tags,
                        project_name=_current_project,
                    )(func)
                    return traced_func(*args, **kwargs)
                # LangSmith puede fallar de formas no previstas por la librería;
                # nunca debe impedir el fallback a logging local.
                except Exception:  # noqa: BLE001, S110
                    pass  # Fallback a logging

            # Fallback: logging a consola
            logger.debug("[TRACE] START  %s  tags=%s", run_name, tags)
            try:
                result = func(*args, **kwargs)
                logger.debug("[TRACE] END    %s  -> %s", run_name, type(result).__name__)
                return result
            except Exception as exc:
                logger.debug("[TRACE] ERROR  %s  -> %s: %s", run_name, type(exc).__name__, exc)
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            run_name = f"{notebook or 'unknown'}/{func.__name__}"
            tags = []
            if notebook:
                tags.append(notebook)
            if concept:
                tags.append(concept)

            if _langsmith_available:
                try:
                    from langsmith import traceable  # type: ignore

                    traced_func = traceable(
                        run_type="chain",
                        name=run_name,
                        tags=tags,
                        project_name=_current_project,
                    )(func)
                    return await traced_func(*args, **kwargs)
                # LangSmith puede fallar de formas no previstas por la librería;
                # nunca debe impedir el fallback a logging local.
                except Exception:  # noqa: BLE001, S110
                    pass  # Fallback a logging

            # Fallback: logging a consola
            logger.debug("[TRACE] START  %s  tags=%s", run_name, tags)
            try:
                result = await func(*args, **kwargs)
                logger.debug("[TRACE] END    %s  -> %s", run_name, type(result).__name__)
                return result
            except Exception as exc:
                logger.debug("[TRACE] ERROR  %s  -> %s: %s", run_name, type(exc).__name__, exc)
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
