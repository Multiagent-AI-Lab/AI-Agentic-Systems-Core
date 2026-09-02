from src.multiagent_core.skills.evaluation import output_scorer


class TestOutputScorer:
    def test_returns_score_result(self):
        result = output_scorer.score_heuristic("Instala numpy con pip install numpy.")
        assert hasattr(result, "score")
        assert hasattr(result, "breakdown")
        assert hasattr(result, "feedback")
        assert hasattr(result, "passed")

    def test_score_in_range(self):
        result = output_scorer.score_heuristic("Texto de prueba con contenido.")
        assert 0.0 <= result.score <= 1.0

    def test_empty_output_low_score(self):
        result = output_scorer.score_heuristic("")
        assert result.score < 0.3

    def test_rich_output_higher_score(self):
        rich = (
            "# Guia de instalacion\n\n"
            "Para instalar el entorno ejecuta:\n\n"
            "```bash\nconda create -n ia_nano python=3.11\nconda activate ia_nano\npip install numpy\n```\n\n"
            "El paquete tiene 3 modulos principales con 42 funciones totales.\n"
            "- Modulo A: procesamiento\n- Modulo B: visualizacion\n- Modulo C: exportacion"
        )
        result = output_scorer.score_heuristic(rich)
        assert result.score > 0.4

    def test_custom_criteria(self):
        criteria = {"length": 0.5, "structure": 0.5}
        result = output_scorer.score_heuristic("texto corto", criteria=criteria)
        assert set(result.breakdown.keys()) == {"length", "structure"}

    def test_llm_none_fallback_to_heuristic(self):
        result = output_scorer.score_with_llm("Texto de prueba.", llm=None)
        assert isinstance(result, output_scorer.ScoreResult)

    def test_passing_threshold(self):
        result = output_scorer.score_heuristic("x", passing_threshold=0.99)
        assert result.passed is False
