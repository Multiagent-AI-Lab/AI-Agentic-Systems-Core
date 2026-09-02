import pytest

from src.multiagent_core.skills.apis import token_budget_guard


class TestTokenBudgetGuard:
    def test_record_call_within_budget(self):
        guard = token_budget_guard.BudgetGuard(budget_usd=10.0, model="gpt-4o-mini")
        guard.record_call(tokens_input=100, tokens_output=50, label="test")
        assert guard.cost_usd > 0
        assert guard.circuit_open is False

    def test_exceeds_budget_opens_circuit(self):
        guard = token_budget_guard.BudgetGuard(budget_usd=0.000001, model="gpt-4o")
        guard.record_call(tokens_input=1000, tokens_output=1000, label="test")
        assert guard.circuit_open is True

    def test_circuit_open_raises(self):
        guard = token_budget_guard.BudgetGuard(budget_usd=0.000001, model="gpt-4o")
        guard.record_call(tokens_input=1000, tokens_output=1000, label="first")
        with pytest.raises(token_budget_guard.BudgetExceededError):
            guard.record_call(tokens_input=1, tokens_output=1, label="second")

    def test_estimate_tokens(self):
        count = token_budget_guard.estimate_tokens("hola mundo esta es una prueba")
        assert count >= 1

    def test_reset_clears_state(self):
        guard = token_budget_guard.BudgetGuard(budget_usd=0.000001, model="gpt-4o")
        guard.record_call(tokens_input=1000, tokens_output=1000, label="test")
        guard.reset()
        assert guard.circuit_open is False
        assert guard.cost_usd == 0.0
