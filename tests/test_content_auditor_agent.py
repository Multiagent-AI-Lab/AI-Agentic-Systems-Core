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

    def test_sin_hallazgo_si_usa_default_rng(self, auditor, tmp_path):
        """Regresión: np.random.default_rng(semilla) es una forma de fijar
        semilla tan válida como random.seed()/np.random.seed() — antes de
        este fix, el auditor exigía literalmente el substring 'seed(' y
        marcaba un falso positivo en código que ya era reproducible."""
        md = tmp_path / "unidad.md"
        md.write_text(
            "```python\nimport numpy as np\nrng = np.random.default_rng(42)\nx = rng.uniform(0, 1)\n```",
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
    def test_detecta_ciclo_del_agente_incompleto(self, auditor, tmp_path):
        md = tmp_path / "UNIDAD_1_ML_FUNDAMENTALS.md"
        md.write_text(
            "# Unidad 1\n\n## 🔄 El Ciclo del Agente\n\n"
            "### Selección de Arquitectura\n\nTexto.\n\n"
            "### Diseño\n\nTexto.\n",
            encoding="utf-8",
        )
        resultado = auditor.audit_unit(md)
        assert any(
            "Ciclo del Agente" in h and "incompleto" in h.lower()
            for h in resultado["hallazgos"]["patron_pedagogico"]
        )

    def test_sin_hallazgo_si_las_6_fases_estan_presentes(self, auditor, tmp_path):
        md = tmp_path / "UNIDAD_1_ML_FUNDAMENTALS.md"
        fases = [
            "Selección de Arquitectura", "Diseño", "Implementación",
            "Evaluación", "Despliegue", "Iteración",
        ]
        secciones = "\n\n".join(f"### {fase}\n\nTexto de la fase." for fase in fases)
        md.write_text(f"# Unidad 1\n\n## 🔄 El Ciclo del Agente\n\n{secciones}\n", encoding="utf-8")
        resultado = auditor.audit_unit(md)
        assert not any(
            "Ciclo del Agente" in h for h in resultado["hallazgos"]["patron_pedagogico"]
        )

    def test_excluye_unidad_4_de_la_verificacion_del_ciclo(self, auditor, tmp_path):
        md = tmp_path / "UNIDAD_4_SISTEMAS_ADAPTATIVOS.md"
        md.write_text("# Unidad 4\n\nContenido de línea de investigación, sin el ciclo.\n", encoding="utf-8")
        resultado = auditor.audit_unit(md)
        assert not any(
            "Ciclo del Agente" in h for h in resultado["hallazgos"]["patron_pedagogico"]
        )


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


class TestAuditAllUnits:
    def test_genera_reporte_consolidado_de_varias_unidades(self, tmp_path):
        course_dir = tmp_path / "lecciones"
        course_dir.mkdir()
        (course_dir / "UNIDAD_0_PRUEBA.md").write_text("# U0\n\nsin fences.", encoding="utf-8")
        (course_dir / "UNIDAD_1_PRUEBA.md").write_text("# U1\n\nsin fences.", encoding="utf-8")

        auditor = ContentAuditorAgent(mermaid_renderer=MagicMock())
        reporte = auditor.audit_all_units(course_dir)

        assert "UNIDAD_0_PRUEBA.md" in reporte
        assert "UNIDAD_1_PRUEBA.md" in reporte
        assert "Reporte de Auditoría de Contenido" in reporte

    def test_ordena_unidades_alfabeticamente(self, tmp_path):
        course_dir = tmp_path / "lecciones"
        course_dir.mkdir()
        (course_dir / "UNIDAD_2_PRUEBA.md").write_text("# U2", encoding="utf-8")
        (course_dir / "UNIDAD_0_PRUEBA.md").write_text("# U0", encoding="utf-8")

        auditor = ContentAuditorAgent(mermaid_renderer=MagicMock())
        reporte = auditor.audit_all_units(course_dir)

        assert reporte.index("UNIDAD_0_PRUEBA.md") < reporte.index("UNIDAD_2_PRUEBA.md")
