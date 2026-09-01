"""
Agente Renderizador de Mermaid (MermaidRenderer)
==================================================

Convierte texto de diagrama Mermaid en un archivo SVG estático, con caché
por hash de contenido, para que los diagramas se vean idénticos en
VS Code, GitHub y Google Colab (que no renderiza ```mermaid``` nativamente).
"""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

SKILL_METADATA = {
    "name": "mermaid_renderer",
    "description": "Renderiza texto Mermaid a SVG estático con caché por hash de contenido.",
    "version": "1.0.0",
    "input": "mermaid_source: str",
    "output": "Path (ruta del .svg generado o cacheado)",
    "requires_api_key": False,
}

_HASH_LENGTH = 16


class MermaidRenderer:
    """Convierte texto Mermaid en SVG, cacheado por hash SHA-256 del contenido."""

    def __init__(self, output_dir: Path) -> None:
        npx_path = shutil.which("npx")
        if npx_path is None:
            raise RuntimeError(
                "Node.js no está instalado o 'npx' no está en el PATH. "
                "Los diagramas Mermaid requieren Node.js para renderizarse como SVG. "
                "Instalar con: winget install OpenJS.NodeJS (Windows) o desde https://nodejs.org"
            )
        self._npx_path = npx_path

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, mermaid_source: str) -> Path:
        """Renderiza texto Mermaid a SVG, reusando el archivo cacheado si existe."""
        file_hash = hashlib.sha256(mermaid_source.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
        svg_path = self.output_dir / f"{file_hash}.svg"
        if svg_path.exists():
            return svg_path

        self._render_con_mmdc(mermaid_source, svg_path)
        return svg_path

    def _render_con_mmdc(self, mermaid_source: str, svg_path: Path) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mmd", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(mermaid_source)
            tmp_path = tmp_file.name
        try:
            cmd = [
                self._npx_path, "--yes", "@mermaid-js/mermaid-cli",
                "-i", tmp_path, "-o", str(svg_path), "-b", "white",
            ]
            ultimo_error = ""
            for intento in range(2):
                resultado = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if resultado.returncode == 0:
                    return
                ultimo_error = resultado.stderr
            raise RuntimeError(ultimo_error)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
