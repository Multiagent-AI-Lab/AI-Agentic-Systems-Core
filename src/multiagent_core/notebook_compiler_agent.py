"""
Agente Compilador de Notebooks (NotebookCompilerAgent)
========================================================

Compila un archivo Markdown (fuente de verdad del contenido pedagógico) a
un Jupyter Notebook (.ipynb), traduciendo encabezados a celdas Markdown y
bloques ```python``` a celdas de código. Usa MathAgent para normalizar
LaTeX y FlowchartAgent para inyectar diagramas automáticos.
"""

from pathlib import Path
from typing import ClassVar

import nbformat as nbf

from ._fence_utils import _scan_fence_segments
from .flowchart_agent import FlowchartAgent

SKILL_METADATA = {
    "name": "notebook_compiler_agent",
    "description": "Compila un Markdown a un Jupyter Notebook (.ipynb).",
    "version": "1.0.0",
    "input": "md_filepath: Path, output_dir: Path",
    "output": "Path (ruta del .ipynb generado)",
    "requires_api_key": False,
}

_MIN_LINEAS_PARA_DIAGRAMA = 5


class MathAgent:
    """Traduce símbolos Unicode matemáticos comunes a comandos LaTeX."""

    _MAPA_SIMBOLOS: ClassVar[dict[str, str]] = {
        "Δ": r"\Delta ",
        "α": r"\alpha ",
        "β": r"\beta ",
        "σ": r"\sigma ",
        "μ": r"\mu ",
        "θ": r"\theta ",
        "∑": r"\sum ",
        "√": r"\sqrt ",
    }

    def process_latex(self, text: str) -> str:
        """Sustituye símbolos Unicode matemáticos por su comando LaTeX equivalente."""
        resultado = text
        for simbolo, latex in self._MAPA_SIMBOLOS.items():
            resultado = resultado.replace(simbolo, latex)
        return resultado


class NotebookCompilerAgent:
    """Agente que traduce un Markdown completo a un archivo .ipynb."""

    def __init__(self) -> None:
        self.math_agent = MathAgent()
        self.flowchart_agent = FlowchartAgent()

    def compile(self, md_filepath: Path, output_dir: Path) -> Path:
        """Compila un archivo Markdown a un notebook .ipynb, preservando el
        orden real de aparición de texto y bloques de código del documento
        fuente."""
        output_dir.mkdir(parents=True, exist_ok=True)
        nb_path = output_dir / (md_filepath.stem + ".ipynb")

        content = md_filepath.read_text(encoding="utf-8")
        nb = nbf.v4.new_notebook()

        for segmento in self._segment_document(content):
            if segmento[0] == "text":
                self._append_text_cells(nb, segmento[1])
            else:
                _, _fence, lang, code = segmento
                self._append_code_segment(nb, lang, code)

        nbf.write(nb, nb_path)
        return nb_path

    def _append_text_cells(self, nb: nbf.NotebookNode, texto: str) -> None:
        """Agrega una celda Markdown por cada párrafo no vacío del texto dado."""
        for seccion in texto.split("\n\n"):
            if not seccion.strip():
                continue
            texto_procesado = self.math_agent.process_latex(seccion)
            nb.cells.append(nbf.v4.new_markdown_cell(texto_procesado))

    def _append_code_segment(self, nb: nbf.NotebookNode, lang: str, code: str) -> None:
        """Agrega la(s) celda(s) correspondientes a un bloque con fence,
        según su lenguaje."""
        if lang == "python":
            nb.cells.append(nbf.v4.new_code_cell(code))
            if code.count("\n") + 1 > _MIN_LINEAS_PARA_DIAGRAMA and "def " in code:
                diagrama = self.flowchart_agent.build_mermaid_flowchart(code)
                if not diagrama.startswith("%%"):
                    nb.cells.append(
                        nbf.v4.new_markdown_cell(f"```mermaid\n{diagrama}\n```")
                    )
        elif lang in ("mermaid", "markdown", "text", ""):
            nb.cells.append(nbf.v4.new_markdown_cell(f"```{lang}\n{code}\n```"))

    def _segment_document(self, content: str) -> list[tuple]:
        """Divide el documento en segmentos ordenados según aparecen en el
        texto fuente: `("text", texto)` para tramos de Markdown y
        `("code", fence, lang, code)` para bloques delimitados por fences.

        Delega el escaneo de fences en `_scan_fence_segments` (compartido
        con `extract_fenced_blocks`, usado por `ContentAuditorAgent`) — una
        sola implementación real del reconocimiento de fences en el repo,
        en vez de reimplementar el mismo bucle en cada consumidor. A
        diferencia de `extract_fenced_blocks` (que descarta el texto
        circundante), este método conserva el texto intercalado entre
        bloques para poder reconstruir el orden real del documento.
        """
        return _scan_fence_segments(content)
