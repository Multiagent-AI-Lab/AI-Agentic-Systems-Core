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

from ._fence_utils import extract_fenced_blocks
from .flowchart_agent import FlowchartAgent
from .mermaid_renderer import MermaidRenderer

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

    def __init__(
        self,
        mermaid_renderer: MermaidRenderer | None = None,
    ) -> None:
        self.math_agent = MathAgent()
        self.flowchart_agent = FlowchartAgent()
        self.mermaid_renderer = mermaid_renderer or MermaidRenderer(
            output_dir=Path.cwd() / "notebooks" / "assets" / "diagramas"
        )

    def compile(self, md_filepath: Path, output_dir: Path) -> Path:
        """Compila un archivo Markdown a un notebook .ipynb."""
        output_dir.mkdir(parents=True, exist_ok=True)
        nb_path = output_dir / (md_filepath.stem + ".ipynb")

        content = md_filepath.read_text(encoding="utf-8")
        nb = nbf.v4.new_notebook()

        bloques = extract_fenced_blocks(content)
        texto_sin_fences = self._quitar_bloques_de_codigo(content, bloques)
        secciones_texto = [s for s in texto_sin_fences.split("\n\n") if s.strip()]

        for seccion in secciones_texto:
            texto_procesado = self.math_agent.process_latex(seccion)
            nb.cells.append(nbf.v4.new_markdown_cell(texto_procesado))

        for fence, lang, code in bloques:
            if lang == "python":
                nb.cells.append(nbf.v4.new_code_cell(code))
                if code.count("\n") + 1 > _MIN_LINEAS_PARA_DIAGRAMA and "def " in code:
                    diagrama = self.flowchart_agent.build_mermaid_flowchart(code)
                    if not diagrama.startswith("%%"):
                        nb.cells.append(
                            nbf.v4.new_markdown_cell(f"```mermaid\n{diagrama}\n```")
                        )
            elif lang in ("mermaid", "markdown", "text", ""):
                nb.cells.append(
                    nbf.v4.new_markdown_cell(f"```{lang}\n{code}\n```")
                )

        nbf.write(nb, nb_path)
        return nb_path

    def _quitar_bloques_de_codigo(self, content: str, bloques: list[tuple]) -> str:
        """Elimina el texto de los bloques con fences del contenido, dejando
        solo el texto Markdown circundante para convertir en celdas de texto."""
        resultado = content
        for fence, lang, code in bloques:
            bloque_completo = f"{fence}{lang}\n{code}\n{fence}"
            resultado = resultado.replace(bloque_completo, "")
        return resultado
