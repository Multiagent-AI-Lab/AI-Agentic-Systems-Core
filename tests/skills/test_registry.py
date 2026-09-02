import pytest

from src.multiagent_core.skills import registry


class TestRegistry:
    def test_discover_skills_returns_list(self):
        skills = registry.discover_skills()
        assert isinstance(skills, list)
        assert len(skills) > 0

    def test_load_skill_context_loader(self):
        mod = registry.load_skill("context_loader")
        assert hasattr(mod, "warm_up")

    def test_load_skill_task_classifier(self):
        mod = registry.load_skill("task_classifier")
        assert hasattr(mod, "classify")

    def test_load_unknown_skill_raises(self):
        with pytest.raises((KeyError, ValueError)):
            registry.load_skill("skill_que_no_existe")
