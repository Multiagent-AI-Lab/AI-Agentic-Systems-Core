"""
Agente Orquestador (OrchestratorAgent)
=========================================

Coordina CodeAuditorAgent, FlowchartAgent, PseudocodeAgent, EvaluatorAgent
y SafetyGateAgent para producir un reporte pedagógico unificado en
Markdown a partir del código o pseudocódigo entregado y el número de
unidad correspondiente. Clasifica automáticamente el tipo de entrada para
evitar invocar agentes que no aplican.
"""

import ast
import re
from pathlib import Path

from .code_auditor_agent import CodeAuditorAgent
from .evaluator_agent import EvaluatorAgent
from .flowchart_agent import FlowchartAgent
from .pseudocode_agent import PseudocodeAgent
from .safety_gate_agent import SafetyGateAgent

SKILL_METADATA = {
    "name": "orchestrator_agent",
    "description": "Coordina CodeAuditorAgent, FlowchartAgent/PseudocodeAgent, EvaluatorAgent y SafetyGateAgent en un reporte pedagógico único, con routing automático por tipo de entrada.",
    "version": "1.0.0",
    "input": "student_code: str, unit_number: int, test_file_path: Optional[Path]",
    "output": "str (reporte Markdown consolidado)",
    "requires_api_key": False,
}

PSEUDOCODE_KEYWORDS = ("INICIO", "FUNCIÓN", "FIN_SI", "FIN_FUNCIÓN", "MIENTRAS", "PARA")
_PSEUDOCODE_KEYWORD_PATTERN = re.compile(r"\b(" + "|".join(PSEUDOCODE_KEYWORDS) + r")\b")


class OrchestratorAgent:
    """Agente que coordina el conjunto de agentes pedagógicos y consolida un reporte único."""

    def __init__(self) -> None:
        self.auditor = CodeAuditorAgent()
        self.flowchart_agent = FlowchartAgent()
        self.pseudocode_agent = PseudocodeAgent()
        self.evaluator = EvaluatorAgent()
        self.safety_gate = SafetyGateAgent()

    def _detect_input_type(self, student_code: str) -> str:
        """Clasifica la entrada para decidir qué sub-agentes invocar.

        Un texto que parsea como Python válido nunca se clasifica como
        pseudocódigo, incluso si contiene palabras clave del pseudocódigo
        dentro de comentarios, docstrings o strings.
        """
        try:
            ast.parse(student_code)
            es_python_valido = True
        except SyntaxError:
            es_python_valido = False

        if not es_python_valido and _PSEUDOCODE_KEYWORD_PATTERN.search(student_code.upper()):
            return "pseudocodigo"
        if "def " in student_code:
            return "python_con_funciones"
        return "python_sin_funciones"

    def generate_pedagogical_report(
        self,
        student_code: str,
        unit_number: int,
        test_file_path: Path | None = None,
    ) -> str:
        """Genera un reporte pedagógico unificado: auditoría + diagrama + calificación."""
        tipo_entrada = self._detect_input_type(student_code)

        secciones = [f"# Reporte Pedagógico Unificado — Unidad {unit_number}", ""]

        if tipo_entrada != "pseudocodigo":
            auditoria = self.auditor.generate_report(student_code, test_file_path)
            secciones.extend([auditoria, ""])

        if tipo_entrada == "pseudocodigo":
            diagrama = self.pseudocode_agent.pseudocode_to_mermaid(student_code)
            secciones.extend([
                "## [DIAGRAMA] Diagrama de Flujo Autogenerado (desde pseudocódigo)",
                "", "```mermaid", diagrama, "```", "",
            ])
        elif tipo_entrada == "python_con_funciones":
            diagrama = self.flowchart_agent.build_mermaid_flowchart(student_code)
            secciones.extend([
                "## [DIAGRAMA] Diagrama de Flujo Autogenerado",
                "", "```mermaid", diagrama, "```", "",
            ])

        evaluacion = self.evaluator.evaluate(student_code, test_file_path)
        secciones.extend([
            "## [CALIFICACIÓN] Evaluación contra Rúbrica Genérica",
            "", evaluacion["retroalimentacion"],
        ])

        reporte = "\n".join(secciones)

        gate_resultado = self.safety_gate.check_output(reporte)
        if not gate_resultado["passed"]:
            alertas = "\n".join(f"- {h}" for h in gate_resultado["hallazgos"])
            reporte += f"\n\n## [SAFETY GATE] Alertas de Guardrails\n\n{alertas}"

        return reporte
