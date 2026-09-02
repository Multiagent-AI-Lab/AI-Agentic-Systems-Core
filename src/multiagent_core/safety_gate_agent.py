"""
Agente de Guardrails (SafetyGateAgent)
=========================================

Valida outputs de agentes de software contra guardrails genéricos: claves
o secretos filtrados, patrones de prompt injection reflejados (el output
del agente repite o cede a una instrucción de "ignora las instrucciones
anteriores"), y formato mínimo esperado (no vacío). Heurístico puro, sin
LLM — en Antigravity-Nano (Sub-proyecto 4) se extiende con validaciones
químicas reales (RDKit, stability_guardian), pero esta versión es
genérica de dominio.
"""

import re
from typing import Any

from ._security_patterns import API_KEY_PATTERN

SKILL_METADATA = {
    "name": "safety_gate_agent",
    "description": "Valida outputs de agentes contra guardrails genéricos (secretos filtrados, prompt injection reflejado, formato mínimo).",
    "version": "1.0.0",
    "input": "agent_output: str",
    "output": "Dict[str, Any] con passed: bool y hallazgos: list[str]",
    "requires_api_key": False,
}

_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"ignora(r)?\s+las?\s+instruccion", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"revela\s+tu", re.IGNORECASE),
)


class SafetyGateAgent:
    """Agente que valida outputs de otros agentes contra guardrails genéricos."""

    def __init__(self) -> None:
        pass

    def check_output(self, agent_output: str) -> dict[str, Any]:
        """Valida un output de agente contra guardrails genéricos.

        Args:
            agent_output: Texto producido por otro agente, a validar.

        Returns:
            Diccionario con "passed" (True si no hay hallazgos) y
            "hallazgos" (lista de descripciones de problemas encontrados).
        """
        hallazgos: list[str] = []

        if not agent_output.strip():
            hallazgos.append("El output está vacío — viola el formato mínimo esperado.")

        for match in API_KEY_PATTERN.finditer(agent_output):
            if "os.environ" not in agent_output[max(0, match.start() - 20):match.end()]:
                hallazgos.append(
                    "Posible credencial o API Key filtrada en texto plano en el output del agente."
                )
                break

        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(agent_output):
                hallazgos.append(
                    "Posible patrón de inyección de instrucciones reflejado en el "
                    "output — el agente pudo haber cedido a instrucciones "
                    "incrustadas en su entrada en vez de en su prompt original."
                )
                break

        return {"passed": len(hallazgos) == 0, "hallazgos": hallazgos}
