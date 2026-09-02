"""src/multiagent_core/skills/apis/token_budget_guard.py
Control de presupuesto de tokens con circuit breaker.
"""

from __future__ import annotations

SKILL_METADATA = {
    "name": "token_budget_guard",
    "domain": "apis",
    "description": "Controla el presupuesto de tokens por sesión con circuit breaker. Soporta múltiples modelos.",
    "version": "1.0.0",
    "input": "tokens_input: int, tokens_output: int, label: str",
    "output": "Actualiza contadores internos. Lanza BudgetExceededError si se supera el presupuesto.",
    "dependencies": [],
}

# Costos en USD por 1M tokens (input/output)
# Fuente: precios aproximados a marzo 2026
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "grok-3": {"input": 3.00, "output": 15.00},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
    "deepseek-r1": {"input": 0.55, "output": 2.19},
    "qwen-2.5-max": {"input": 0.40, "output": 1.20},
    "qwen-2.5-plus": {"input": 0.10, "output": 0.40},
    "ollama": {"input": 0.00, "output": 0.00},
    "default": {"input": 0.50, "output": 1.50},
}


class BudgetExceededError(Exception):
    """Se lanza cuando el circuit breaker está abierto por presupuesto superado."""


def estimate_tokens(text: str) -> int:
    """
    Estima el número de tokens en un texto.
    Aproximación: 1 token ≈ 4 caracteres (heurística BPE estándar).

    Args:
        text: Texto a contar.

    Returns:
        Número estimado de tokens (siempre >= 1).
    """
    if not text:
        return 1
    return max(1, len(text) // 4)


class BudgetGuard:
    """
    Circuit breaker para controlar el consumo de tokens y costo USD.

    Attributes:
        tokens_input (int): Tokens de input acumulados en la sesión.
        tokens_output (int): Tokens de output acumulados.
        cost_usd (float): Costo total estimado en USD.
        circuit_open (bool): True cuando el presupuesto fue superado.
    """

    def __init__(self, budget_usd: float, model: str = "gemini-2.0-flash") -> None:
        """
        Args:
            budget_usd: Presupuesto máximo en USD para esta sesión.
            model: Nombre del modelo para calcular costos. Ver MODEL_COSTS.
        """
        self.budget_usd = budget_usd
        self.model = model
        self._costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])

        self.tokens_input: int = 0
        self.tokens_output: int = 0
        self.cost_usd: float = 0.0
        self.circuit_open: bool = False
        self._call_log: list[dict] = []

    def record_call(
        self,
        tokens_input: int,
        tokens_output: int,
        label: str = "",
    ) -> None:
        """
        Registra una llamada al LLM y actualiza los contadores.

        Args:
            tokens_input: Tokens enviados al modelo.
            tokens_output: Tokens recibidos del modelo.
            label: Etiqueta descriptiva para el log.

        Raises:
            BudgetExceededError: Si el circuit breaker está abierto (presupuesto ya superado).
        """
        if self.circuit_open:
            raise BudgetExceededError(
                f"Presupuesto de ${self.budget_usd:.4f} USD superado. "
                f"Costo acumulado: ${self.cost_usd:.4f} USD. "
                "Usa reset() para iniciar nueva sesión."
            )

        call_cost = (tokens_input / 1_000_000) * self._costs["input"] + (
            tokens_output / 1_000_000
        ) * self._costs["output"]

        self.tokens_input += tokens_input
        self.tokens_output += tokens_output
        self.cost_usd += call_cost

        self._call_log.append(
            {
                "label": label,
                "tokens_in": tokens_input,
                "tokens_out": tokens_output,
                "cost": call_cost,
                "cumulative_cost": self.cost_usd,
            }
        )

        if self.cost_usd >= self.budget_usd:
            self.circuit_open = True

    def reset(self) -> None:
        """Reinicia todos los contadores y abre el circuit breaker."""
        self.tokens_input = 0
        self.tokens_output = 0
        self.cost_usd = 0.0
        self.circuit_open = False
        self._call_log = []

    def report(self) -> str:
        """Retorna un reporte legible del consumo actual de la sesión."""
        lines = [
            f"--- BudgetGuard Report ({self.model}) ---",
            f"  Tokens input:   {self.tokens_input:,}",
            f"  Tokens output:  {self.tokens_output:,}",
            f"  Costo total:    ${self.cost_usd:.6f} USD",
            f"  Presupuesto:    ${self.budget_usd:.4f} USD",
            f"  Uso del budget: {(self.cost_usd / self.budget_usd * 100):.1f}%",
            f"  Circuit:        {'ABIERTO (budget superado)' if self.circuit_open else 'cerrado (OK)'}",
            f"  Llamadas:       {len(self._call_log)}",
        ]
        return "\n".join(lines)
