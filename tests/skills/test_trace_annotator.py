import pytest

from src.multiagent_core.skills.observability import trace_annotator


class TestTraceAnnotator:
    def test_traced_decorator(self):
        @trace_annotator.traced(notebook="test_nb", concept="test_concept")
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_traced_captures_exception(self):
        @trace_annotator.traced(notebook="test_nb", concept="error_concept")
        def failing_func():
            raise ValueError("error de prueba")

        with pytest.raises(ValueError):
            failing_func()
