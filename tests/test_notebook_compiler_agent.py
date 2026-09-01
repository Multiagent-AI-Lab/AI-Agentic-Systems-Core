"""Tests de caracterización para NotebookCompilerAgent."""

from pathlib import Path
from unittest.mock import MagicMock

import nbformat
import pytest

from src.multiagent_core.notebook_compiler_agent import MathAgent, NotebookCompilerAgent


@pytest.fixture
def mock_renderer():
    renderer = MagicMock()
    renderer.render.return_value = Path("/tmp/fake.svg")
    return renderer


@pytest.fixture
def compiler(mock_renderer) -> NotebookCompilerAgent:
    return NotebookCompilerAgent(mermaid_renderer=mock_renderer)


class TestMathAgent:
    def test_traduce_delta_a_latex(self):
        math_agent = MathAgent()
        resultado = math_agent.process_latex("El símbolo Δ representa el cambio")
        assert r"\Delta" in resultado or "Δ" not in resultado


class TestCompile:
    def test_genera_archivo_ipynb_valido(self, compiler: NotebookCompilerAgent, tmp_path: Path):
        md_file = tmp_path / "UNIDAD_TEST.md"
        md_file.write_text("# Título\n\nContenido de prueba.\n", encoding="utf-8")
        output_dir = tmp_path / "notebooks"

        resultado = compiler.compile(md_file, output_dir)

        assert resultado.exists()
        assert resultado.suffix == ".ipynb"
        nb = nbformat.read(resultado, as_version=4)
        assert len(nb.cells) > 0

    def test_bloque_python_se_convierte_en_celda_de_codigo(
        self, compiler: NotebookCompilerAgent, tmp_path: Path
    ):
        md_file = tmp_path / "UNIDAD_TEST.md"
        md_file.write_text(
            "# Título\n\n```python\nx = 1\n```\n", encoding="utf-8"
        )
        output_dir = tmp_path / "notebooks"

        resultado = compiler.compile(md_file, output_dir)
        nb = nbformat.read(resultado, as_version=4)

        celdas_codigo = [c for c in nb.cells if c.cell_type == "code"]
        assert any("x = 1" in c.source for c in celdas_codigo)

    def test_crea_output_dir_si_no_existe(self, compiler: NotebookCompilerAgent, tmp_path: Path):
        md_file = tmp_path / "UNIDAD_TEST.md"
        md_file.write_text("# Título\n", encoding="utf-8")
        output_dir = tmp_path / "notebooks_nuevo"

        compiler.compile(md_file, output_dir)

        assert output_dir.exists()
