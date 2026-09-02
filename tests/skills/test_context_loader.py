import pytest

from src.multiagent_core.skills.agent_warmup import context_loader


class TestContextLoader:
    def test_warm_up_returns_required_keys(self):
        result = context_loader.warm_up("research")
        assert "system_context" in result
        assert "messages" in result
        assert "domain" in result

    def test_domain_is_preserved(self):
        for domain in ["research", "engineering", "teaching", "nanotechnology", "data_analysis"]:
            result = context_loader.warm_up(domain)
            assert result["domain"] == domain

    def test_messages_have_system_role(self):
        result = context_loader.warm_up("nanotechnology")
        assert result["messages"][0]["role"] == "system"

    def test_custom_domain_requires_config(self):
        with pytest.raises(ValueError):
            context_loader.warm_up("custom")

    def test_custom_domain_with_config(self):
        result = context_loader.warm_up("custom", {"system_prompt": "Eres un experto."})
        assert result["system_context"] == "Eres un experto."

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError):
            context_loader.warm_up("dominio_inexistente")

    def test_list_available_domains(self):
        domains = context_loader.list_available_domains()
        assert "research" in domains
        assert "custom" in domains
        assert len(domains) >= 6

    def test_apply_to_agent_with_system_message(self):
        class FakeAgent:
            system_message = ""

        agent = context_loader.apply_to_agent(FakeAgent(), "teaching")
        assert len(agent.system_message) > 10

    def test_apply_to_agent_with_backstory(self):
        class FakeAgent:
            backstory = "Experto inicial."

        agent = context_loader.apply_to_agent(FakeAgent(), "engineering")
        assert "Experto inicial." in agent.backstory
