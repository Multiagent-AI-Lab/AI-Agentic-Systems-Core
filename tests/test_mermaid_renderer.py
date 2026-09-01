"""Tests TDD para MermaidRenderer (texto Mermaid -> SVG con caché)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.multiagent_core.mermaid_renderer import MermaidRenderer


class TestMermaidRendererInit:
    def test_lanza_error_si_npx_no_esta_en_path(self, tmp_path: Path):
        with patch("src.multiagent_core.mermaid_renderer.shutil.which") as mock_which:
            mock_which.return_value = None
            with pytest.raises(RuntimeError, match="Node.js"):
                MermaidRenderer(output_dir=tmp_path)

    def test_crea_output_dir_si_no_existe(self, tmp_path: Path):
        output_dir = tmp_path / "diagramas"
        with patch("src.multiagent_core.mermaid_renderer.shutil.which", return_value="/usr/bin/npx"):
            MermaidRenderer(output_dir=output_dir)
        assert output_dir.exists()


class TestRender:
    def test_genera_svg_invocando_mmdc(self, tmp_path: Path):
        with patch("src.multiagent_core.mermaid_renderer.shutil.which", return_value="/usr/bin/npx"):
            renderer = MermaidRenderer(output_dir=tmp_path)
        with patch(
            "src.multiagent_core.mermaid_renderer.subprocess.run"
        ) as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            svg_path = renderer.render("graph TD\n    A --> B")
        assert svg_path.suffix == ".svg"
        mock_run.assert_called_once()

    def test_reusa_cache_si_el_svg_ya_existe(self, tmp_path: Path):
        """subprocess.run está mockeado y no crea el .svg de verdad (eso lo
        hace mmdc en producción) -- para probar la ruta de caché real, el
        mock debe escribir el archivo él mismo la primera vez, simulando
        lo que mmdc haría, antes de que la segunda llamada a render()
        pueda encontrarlo con svg_path.exists()."""
        with patch("src.multiagent_core.mermaid_renderer.shutil.which", return_value="/usr/bin/npx"):
            renderer = MermaidRenderer(output_dir=tmp_path)

        def fake_run(cmd, **kwargs):
            svg_path = Path(cmd[cmd.index("-o") + 1])
            svg_path.write_text("<svg></svg>", encoding="utf-8")
            resultado = MagicMock()
            resultado.returncode = 0
            resultado.stderr = ""
            return resultado

        with patch(
            "src.multiagent_core.mermaid_renderer.subprocess.run", side_effect=fake_run
        ) as mock_run:
            primera = renderer.render("graph TD\n    A --> B")
            segunda = renderer.render("graph TD\n    A --> B")
        mock_run.assert_called_once()  # Solo la primera llamada invoca mmdc.
        assert primera == segunda
        assert primera.exists() is True

    def test_lanza_error_si_mmdc_falla_las_dos_veces(self, tmp_path: Path):
        with patch("src.multiagent_core.mermaid_renderer.shutil.which", return_value="/usr/bin/npx"):
            renderer = MermaidRenderer(output_dir=tmp_path)
        with patch("src.multiagent_core.mermaid_renderer.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Parse error"
            with pytest.raises(RuntimeError, match="Parse error"):
                renderer.render("graph TD\n    A --> B")
        assert mock_run.call_count == 2  # Reintenta una vez antes de fallar.
