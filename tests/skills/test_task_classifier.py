from src.multiagent_core.skills.routing import task_classifier


class TestTaskClassifier:
    def test_returns_required_keys(self):
        result = task_classifier.classify("analiza este dataset de nanoparticulas")
        for key in ("category", "recommended_agent", "confidence", "reasoning", "all_scores"):
            assert key in result

    def test_data_analysis_routing(self):
        result = task_classifier.classify(
            "analiza el dataset y visualiza los datos con estadistica"
        )
        assert result["category"] == "data_analysis"
        assert result["confidence"] > 0.0

    def test_code_generation_routing(self):
        result = task_classifier.classify("genera una funcion de Python para calcular el RMSE")
        assert result["category"] == "code_generation"

    def test_unknown_task_low_confidence(self):
        result = task_classifier.classify("xyz abc def")
        assert result["category"] == "unknown"
        assert result["confidence"] < 0.15

    def test_available_agents_constraint(self):
        result = task_classifier.classify(
            "escribe el codigo para preprocesar datos",
            available_agents=["my_code_agent", "my_data_agent"],
        )
        assert result["recommended_agent"] in ["my_code_agent", "my_data_agent"]

    def test_classify_task_alias(self):
        assert task_classifier.classify_task is task_classifier.classify

    def test_all_scores_have_all_categories(self):
        result = task_classifier.classify("investiga el estado del arte")
        for cat in task_classifier.ROUTING_RULES:
            assert cat in result["all_scores"]
