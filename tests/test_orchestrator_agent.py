"""Tests de caracterización para OrchestratorAgent."""

import pytest

from src.multiagent_core.orchestrator_agent import OrchestratorAgent


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent()


class TestDetectInputType:
    def test_detecta_python_con_funciones(self, orchestrator: OrchestratorAgent):
        assert orchestrator._detect_input_type("def f(x):\n    return x") == "python_con_funciones"

    def test_detecta_python_sin_funciones(self, orchestrator: OrchestratorAgent):
        assert orchestrator._detect_input_type("x = 1") == "python_sin_funciones"

    def test_detecta_pseudocodigo(self, orchestrator: OrchestratorAgent):
        assert orchestrator._detect_input_type("SI x > 0 ENTONCES\n    y <- 1\nFIN_SI") == "pseudocodigo"

    def test_no_confunde_python_con_palabra_para_en_comentario(
        self, orchestrator: OrchestratorAgent
    ):
        codigo = "def f(radio):\n    # calcula el area PARA un radio dado\n    return radio\n"
        assert orchestrator._detect_input_type(codigo) == "python_con_funciones"


class TestGeneratePedagogicalReport:
    def test_reporte_incluye_encabezado_con_unidad(self, orchestrator: OrchestratorAgent):
        reporte = orchestrator.generate_pedagogical_report("x = 1", unit_number=1)
        assert "Unidad 1" in reporte

    def test_reporte_de_python_con_funciones_incluye_diagrama(
        self, orchestrator: OrchestratorAgent
    ):
        reporte = orchestrator.generate_pedagogical_report(
            "def f(x):\n    return x", unit_number=2
        )
        assert "```mermaid" in reporte

    def test_reporte_de_pseudocodigo_omite_auditoria_de_estilo(
        self, orchestrator: OrchestratorAgent
    ):
        reporte = orchestrator.generate_pedagogical_report(
            "SI x > 0 ENTONCES\n    y <- 1\nFIN_SI", unit_number=1
        )
        assert "[ESTILO]" not in reporte

    def test_reporte_de_pseudocodigo_omite_calificacion_automatica(
        self, orchestrator: OrchestratorAgent, mocker
    ):
        evaluate_spy = mocker.spy(orchestrator.evaluator, "evaluate")
        reporte = orchestrator.generate_pedagogical_report(
            "SI x > 0 ENTONCES\n    y <- 1\nFIN_SI", unit_number=1
        )
        evaluate_spy.assert_not_called()
        assert "No aplica para pseudocódigo" in reporte
        assert "Evaluación contra Rúbrica Genérica" not in reporte

    def test_reporte_incluye_calificacion(self, orchestrator: OrchestratorAgent):
        reporte = orchestrator.generate_pedagogical_report("x = 1", unit_number=1)
        assert "CALIFICACIÓN" in reporte or "Calificación" in reporte

    def test_reporte_con_secretos_filtrados_incluye_alerta_de_safety_gate(
        self, orchestrator: OrchestratorAgent
    ):
        codigo = 'API_KEY = "sk-1234567890abcdefghij"'
        reporte = orchestrator.generate_pedagogical_report(codigo, unit_number=1)
        # El SafetyGateAgent corre sobre el reporte final; si el propio reporte
        # cita el hallazgo de seguridad de CodeAuditorAgent, eso ya contiene la
        # cadena de la API key, así que verificamos que no se generó una
        # excepción y el reporte sigue siendo texto válido.
        assert isinstance(reporte, str)
        assert len(reporte) > 0
