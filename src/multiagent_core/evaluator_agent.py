"""
Agente Evaluador (EvaluatorAgent)
====================================

Califica código contra una rúbrica genérica de 5 criterios, reutilizando
CodeAuditorAgent para el análisis de estilo, seguridad y pruebas. El
contenido exacto de la rúbrica (nombres finales, descripciones por nivel)
se redacta en el Sub-proyecto 3 de contenido pedagógico.
"""

import ast
from pathlib import Path
from typing import Any

from .code_auditor_agent import CodeAuditorAgent

SKILL_METADATA = {
    "name": "evaluator_agent",
    "description": "Califica código contra una rúbrica genérica de 5 criterios.",
    "version": "1.0.0",
    "input": "student_code: str, test_file_path: Optional[Path]",
    "output": "Dict[str, Any] con criterios calificados y calificación final",
    "requires_api_key": False,
}

NIVELES = ("Insuficiente", "En desarrollo", "Competente", "Sobresaliente")
PUNTAJE_POR_NIVEL = {
    "Insuficiente": 45.0,
    "En desarrollo": 67.0,
    "Competente": 82.0,
    "Sobresaliente": 95.0,
}


class EvaluatorAgent:
    """Agente que evalúa código contra la rúbrica genérica del curso."""

    def __init__(self):
        self.auditor = CodeAuditorAgent()

    def _evaluar_correccion_logica(self, student_code: str) -> dict[str, str]:
        try:
            ast.parse(student_code)
        except SyntaxError as e:
            return {"nivel": "Insuficiente", "retroalimentacion": f"El código no compila: {e}"}
        return {
            "nivel": "Competente",
            "retroalimentacion": "El código es sintácticamente válido. Verifica manualmente casos borde.",
        }

    def _evaluar_proceso(self, student_code: str) -> dict[str, str]:
        """Evalúa el criterio del patrón pedagógico central del curso.

        No verificable automáticamente a partir del código fuente solo;
        siempre retorna "En desarrollo" con una nota para el estudiante.
        """
        return {
            "nivel": "En desarrollo",
            "retroalimentacion": (
                "No es posible verificar automáticamente el pseudocódigo ni el "
                "diagrama previos; documenta el ciclo completo del patrón "
                "pedagógico del curso en tu entrega."
            ),
        }

    def _evaluar_calidad_codigo(self, student_code: str) -> dict[str, str]:
        style_issues = self.auditor.audit_style(student_code)
        security_issues = self.auditor.audit_security(student_code)
        total_issues = len(style_issues) + len(security_issues)

        if security_issues:
            nivel = "Insuficiente"
        elif total_issues >= 2:
            nivel = "En desarrollo"
        elif total_issues == 1:
            nivel = "Competente"
        else:
            nivel = "Sobresaliente"

        detalle = style_issues + security_issues
        retro = "Hallazgos: " + "; ".join(detalle) if detalle else "Sin hallazgos de estilo ni seguridad."
        return {"nivel": nivel, "retroalimentacion": retro}

    def _evaluar_reproducibilidad(self, student_code: str) -> dict[str, str]:
        usa_aleatoriedad = "random." in student_code or "random(" in student_code
        if not usa_aleatoriedad:
            return {
                "nivel": "Sobresaliente",
                "retroalimentacion": "El código es determinista; no requiere semilla fija.",
            }

        usa_seed = "seed(" in student_code
        if usa_seed:
            return {
                "nivel": "Competente",
                "retroalimentacion": "Usa aleatoriedad con una semilla fija (seed); el resultado es reproducible entre ejecuciones.",
            }

        return {
            "nivel": "Insuficiente",
            "retroalimentacion": (
                "Usa aleatoriedad (random/np.random) sin fijar una semilla (seed); "
                "el resultado cambiará en cada ejecución, dificultando la verificación y la depuración."
            ),
        }

    def _evaluar_pruebas(self, test_file_path: Path | None) -> dict[str, str]:
        if test_file_path is None or not Path(test_file_path).exists():
            return {"nivel": "Insuficiente", "retroalimentacion": "No se proporcionó un archivo de pruebas pytest."}

        resultado = self.auditor.run_pytest(Path(test_file_path))
        if resultado["passed"]:
            return {"nivel": "Competente", "retroalimentacion": "Las pruebas unitarias proporcionadas pasan correctamente."}
        return {"nivel": "En desarrollo", "retroalimentacion": f"Las pruebas fallan:\n{resultado['output']}"}

    def evaluate(
        self, student_code: str, test_file_path: Path | None = None
    ) -> dict[str, Any]:
        """Evalúa el código contra los 5 criterios de la rúbrica genérica."""
        criterios = {
            "Corrección lógica": self._evaluar_correccion_logica(student_code),
            "Proceso (patrón pedagógico)": self._evaluar_proceso(student_code),
            "Calidad de código": self._evaluar_calidad_codigo(student_code),
            "Pruebas (pytest)": self._evaluar_pruebas(test_file_path),
            "Reproducibilidad": self._evaluar_reproducibilidad(student_code),
        }

        puntajes = [PUNTAJE_POR_NIVEL[c["nivel"]] for c in criterios.values()]
        calificacion_final = round(sum(puntajes) / len(puntajes), 2)

        resumen = "\n".join(
            f"- {nombre}: {c['nivel']} — {c['retroalimentacion']}"
            for nombre, c in criterios.items()
        )
        retroalimentacion = f"Calificación final: {calificacion_final}/100\n\n{resumen}"

        return {
            "criterios": criterios,
            "calificacion_final": calificacion_final,
            "retroalimentacion": retroalimentacion,
        }
