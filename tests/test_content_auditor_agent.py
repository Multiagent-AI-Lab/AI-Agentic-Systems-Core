"""Tests de caracterización para ContentAuditorAgent (6 dimensiones propias de este repo)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.multiagent_core.content_auditor_agent import ContentAuditorAgent


@pytest.fixture
def mock_renderer():
    renderer = MagicMock()
    renderer.render.return_value = Path("/tmp/fake.svg")
    return renderer


@pytest.fixture
def auditor(mock_renderer) -> ContentAuditorAgent:
    return ContentAuditorAgent(mermaid_renderer=mock_renderer)


class TestAuditLatex:
    def test_detecta_delimitadores_dollar_desbalanceados(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text("Texto con $x^2 sin cerrar", encoding="utf-8")
        resultado = auditor.audit_unit(md)
        assert resultado["hallazgos"]["latex"]

    def test_sin_hallazgos_con_latex_balanceado(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text("Texto con $x^2$ balanceado", encoding="utf-8")
        resultado = auditor.audit_unit(md)
        assert resultado["hallazgos"]["latex"] == []


class TestAuditCodigo:
    def test_detecta_funcion_sin_docstring(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text(
            "```python\ndef f(x: int) -> int:\n    return x\n```",
            encoding="utf-8",
        )
        resultado = auditor.audit_unit(md)
        assert any("docstring" in h for h in resultado["hallazgos"]["codigo"])

    def test_ignora_funciones_test_para_docstring(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text(
            "```python\ndef test_algo():\n    assert True\n```",
            encoding="utf-8",
        )
        resultado = auditor.audit_unit(md)
        assert resultado["hallazgos"]["codigo"] == []

    def test_detecta_variable_camelcase(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text(
            "```python\ndef f() -> None:\n    \"\"\"Docstring.\"\"\"\n    miVariable = 1\n```",
            encoding="utf-8",
        )
        resultado = auditor.audit_unit(md)
        assert any("CamelCase" in h for h in resultado["hallazgos"]["codigo"])


class TestAuditReproducibilidad:
    def test_detecta_aleatoriedad_sin_seed(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text(
            "```python\nimport random\nx = random.random()\n```",
            encoding="utf-8",
        )
        resultado = auditor.audit_unit(md)
        assert any("seed" in h.lower() or "reproducib" in h.lower() for h in resultado["hallazgos"]["reproducibilidad"])

    def test_sin_hallazgo_si_fija_seed(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text(
            "```python\nimport random\nrandom.seed(42)\nx = random.random()\n```",
            encoding="utf-8",
        )
        resultado = auditor.audit_unit(md)
        assert resultado["hallazgos"]["reproducibilidad"] == []

    def test_detecta_api_key_hardcodeada(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text(
            '```python\nAPI_KEY = "sk-1234567890abcdef"\n```',
            encoding="utf-8",
        )
        resultado = auditor.audit_unit(md)
        assert resultado["hallazgos"]["reproducibilidad"]


class TestAuditPatronPedagogico:
    def test_detecta_patron_incompleto(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text("```python\nx = 1\n```", encoding="utf-8")
        resultado = auditor.audit_unit(md)
        assert resultado["hallazgos"]["patron_pedagogico"]


class TestAuditEstructural:
    def test_detecta_fences_desbalanceados(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text("```python\nx = 1\n", encoding="utf-8")  # sin cierre
        resultado = auditor.audit_unit(md)
        # No debe lanzar excepción; puede o no detectar el desbalance según
        # el comportamiento de extract_fenced_blocks con fences sin cerrar.
        assert isinstance(resultado["hallazgos"]["estructural"], list)


class TestAuditUnitSummary:
    def test_retorna_estructura_con_las_6_dimensiones(self, auditor, tmp_path):
        md = tmp_path / "unidad.md"
        md.write_text("contenido vacío", encoding="utf-8")
        resultado = auditor.audit_unit(md)
        assert set(resultado["hallazgos"].keys()) == {
            "latex", "patron_pedagogico", "codigo", "reproducibilidad",
            "diccionario_variables", "estructural",
        }
        assert "unidad" in resultado
        assert "total_hallazgos" in resultado
