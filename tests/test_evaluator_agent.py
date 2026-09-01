"""Tests de caracterización para EvaluatorAgent."""

from pathlib import Path

import pytest

from src.multiagent_core.evaluator_agent import EvaluatorAgent


@pytest.fixture
def evaluator() -> EvaluatorAgent:
    return EvaluatorAgent()


class TestEvaluate:
    def test_retorna_los_5_criterios(self, evaluator: EvaluatorAgent):
        resultado = evaluator.evaluate("tasa = 0.01")
        assert len(resultado["criterios"]) == 5
        assert "Corrección lógica" in resultado["criterios"]
        assert "Calidad de código" in resultado["criterios"]
        assert "Pruebas (pytest)" in resultado["criterios"]
        assert "Reproducibilidad" in resultado["criterios"]

    def test_codigo_con_error_de_sintaxis_es_insuficiente_en_correccion(
        self, evaluator: EvaluatorAgent
    ):
        resultado = evaluator.evaluate("def f(:\n    pass")
        assert resultado["criterios"]["Corrección lógica"]["nivel"] == "Insuficiente"

    def test_codigo_sin_aleatoriedad_es_sobresaliente_en_reproducibilidad(
        self, evaluator: EvaluatorAgent
    ):
        resultado = evaluator.evaluate("x = 1 + 1")
        assert resultado["criterios"]["Reproducibilidad"]["nivel"] == "Sobresaliente"

    def test_aleatoriedad_sin_seed_es_insuficiente_en_reproducibilidad(
        self, evaluator: EvaluatorAgent
    ):
        resultado = evaluator.evaluate("import random\nx = random.random()")
        assert resultado["criterios"]["Reproducibilidad"]["nivel"] == "Insuficiente"

    def test_sin_test_file_es_insuficiente_en_pruebas(self, evaluator: EvaluatorAgent):
        resultado = evaluator.evaluate("x = 1")
        assert resultado["criterios"]["Pruebas (pytest)"]["nivel"] == "Insuficiente"

    def test_con_test_file_que_pasa_es_competente_en_pruebas(
        self, evaluator: EvaluatorAgent, tmp_path: Path
    ):
        test_file = tmp_path / "test_dummy.py"
        test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        resultado = evaluator.evaluate("x = 1", test_file_path=test_file)
        assert resultado["criterios"]["Pruebas (pytest)"]["nivel"] == "Competente"

    def test_calificacion_final_es_promedio_de_los_5_criterios(
        self, evaluator: EvaluatorAgent
    ):
        resultado = evaluator.evaluate("x = 1")
        assert 0 <= resultado["calificacion_final"] <= 100

    def test_retroalimentacion_incluye_calificacion_final(self, evaluator: EvaluatorAgent):
        resultado = evaluator.evaluate("x = 1")
        assert "Calificación final" in resultado["retroalimentacion"]
