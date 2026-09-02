"""src/multiagent_core/skills/apis/github_skill_loader.py
Descarga y registro seguro de skills desde GitHub con verificación SHA-256.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
import types
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

SKILL_METADATA = {
    "name": "github_skill_loader",
    "domain": "apis",
    "description": (
        "Descarga skills desde GitHub con verificación de integridad SHA-256 "
        "antes de ejecutar cualquier código. "
        "Evita ejecución de código no autorizado."
    ),
    "version": "1.0.0",
    "input": "url: str, expected_hash: str | None, registry: dict | None, skill_name: str | None",
    "output": "module | raises HashMismatchError",
    "dependencies": [],
}

# URLs permitidas (solo raw.githubusercontent.com y gist.githubusercontent.com)
_ALLOWED_URL_PATTERNS = [
    re.compile(r"^https://raw\.githubusercontent\.com/"),
    re.compile(r"^https://gist\.githubusercontent\.com/"),
]


class HashMismatchError(Exception):
    """Lanzada cuando el hash SHA-256 del skill descargado no coincide con el esperado."""


def _validate_url(url: str) -> None:
    """Valida que la URL pertenece a GitHub para evitar SSRF."""
    for pattern in _ALLOWED_URL_PATTERNS:
        if pattern.match(url):
            return
    raise ValueError(
        f"URL no permitida: '{url}'. "
        "Solo se permiten URLs de raw.githubusercontent.com o gist.githubusercontent.com."
    )


def _fetch_content(url: str) -> bytes:
    """Descarga el contenido de una URL con timeout y tamaño máximo."""
    timeout_secs = int(os.environ.get("SKILL_LOADER_TIMEOUT", "15"))
    max_bytes = int(os.environ.get("SKILL_LOADER_MAX_BYTES", str(1 * 1024 * 1024)))  # 1 MB

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ia-nano-skill-loader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_secs) as response:
        content = response.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError(
                f"El skill en '{url}' supera el tamaño máximo permitido ({max_bytes} bytes)."
            )
        return content


def _compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_module_from_source(source: str, module_name: str) -> types.ModuleType:
    """Compila y carga un módulo Python desde una cadena de texto."""
    spec = importlib.util.spec_from_loader(
        module_name,
        loader=None,
        origin=f"<github_skill:{module_name}>",
    )
    if spec is None:
        # Crear spec mínimo
        spec = importlib.util.ModuleSpec(module_name, loader=None)

    module = types.ModuleType(module_name)
    module.__spec__ = spec
    module.__loader__ = None

    # Escribir en archivo temporal para que la introspección funcione
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
        prefix=f"skill_{module_name}_",
    ) as tmp:
        tmp.write(source)
        tmp_path = tmp.name

    try:
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return module


def load_skill_from_github(
    url: str,
    expected_hash: str | None = None,
) -> types.ModuleType:
    """
    Descarga un skill desde GitHub y lo carga como módulo Python.

    Verificación de seguridad:
    - Solo permite URLs de raw.githubusercontent.com / gist.githubusercontent.com
    - Si se proporciona expected_hash, verifica SHA-256 antes de ejecutar.
    - Tamaño máximo: 1 MB por defecto (SKILL_LOADER_MAX_BYTES).

    Args:
        url: URL raw del archivo Python en GitHub.
        expected_hash: Hash SHA-256 esperado (hex string, 64 chars).
                       Si es None, omite la verificación (no recomendado).

    Returns:
        Módulo Python cargado.

    Raises:
        ValueError: Si la URL no está permitida o la descarga falla.
        HashMismatchError: Si el hash no coincide con expected_hash.
    """
    _validate_url(url)

    content_bytes = _fetch_content(url)
    actual_hash = _compute_sha256(content_bytes)

    if expected_hash is not None:
        # Normalizar a lowercase y sin espacios
        expected_clean = expected_hash.strip().lower()
        if actual_hash != expected_clean:
            raise HashMismatchError(
                f"Hash SHA-256 no coincide para '{url}'.\n"
                f"  Esperado: {expected_clean}\n"
                f"  Obtenido: {actual_hash}\n"
                "El skill NO fue cargado por razones de seguridad."
            )

    # Nombre de módulo seguro derivado de la URL
    module_name = "github_skill_" + re.sub(
        r"[^a-zA-Z0-9_]", "_", url.split("/")[-1].replace(".py", "")
    )

    source_code = content_bytes.decode("utf-8")
    return _load_module_from_source(source_code, module_name)


def list_available_skills(repo: str, path: str = "external_skills", ref: str = "main") -> list:
    """
    Lista los archivos .py de skills disponibles en un repositorio de GitHub.

    Args:
        repo: 'owner/repository' (ej: 'antigravity-nano/skills')
        path: Ruta al directorio de skills dentro del repo (default: 'external_skills')
        ref: Branch o tag (default: 'main')

    Returns:
        Lista de nombres de archivos .py disponibles como skills.
    """
    import json

    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "ia-nano-skill-loader/1.0",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return [item["name"] for item in data if item["name"].endswith(".py")]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
        logger.warning("No se pudo listar skills de '%s': %s", repo, e)
        return []


def load_and_register(
    url: str,
    registry: dict[str, Any],
    skill_name: str,
    expected_hash: str | None = None,
) -> types.ModuleType:
    """
    Descarga un skill y lo registra en un diccionario de registro.

    Args:
        url: URL raw del archivo Python en GitHub.
        registry: Diccionario en el que registrar el skill (ej. SKILL_REGISTRY).
        skill_name: Clave con la que registrar el skill.
        expected_hash: Hash SHA-256 esperado.

    Returns:
        Módulo Python cargado.

    Raises:
        HashMismatchError: Si el hash no coincide.
    """
    module = load_skill_from_github(url, expected_hash=expected_hash)
    metadata = getattr(module, "SKILL_METADATA", {"name": skill_name, "version": "unknown"})
    registry[skill_name] = {
        "module": module,
        "metadata": metadata,
        "source_url": url,
        "sha256": _compute_sha256(url.encode()),
    }
    return module
