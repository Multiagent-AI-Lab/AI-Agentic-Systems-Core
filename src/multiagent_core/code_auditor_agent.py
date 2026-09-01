"""
Agente Auditor de Código (CodeAuditorAgent)
=============================================

Audita la calidad y la seguridad del código Python:
- Formato y estilo PEP 8.
- Ejecución de pruebas unitarias automáticas (pytest).
- Análisis estático de seguridad (higiene OWASP y riesgos de código
  generado por IA).
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SKILL_METADATA = {
    "name": "code_auditor_agent",
    "description": "Audita estilo PEP8 y seguridad OWASP de código Python.",
    "version": "1.0.0",
    "input": "code: str, test_file_path: Optional[Path]",
    "output": "List[str] (audit_style/audit_security) | str (generate_report)",
    "requires_api_key": False,
}


class CodeAuditorAgent:
    """Agente que analiza código buscando fallas de estilo y de seguridad."""

    def __init__(self):
        pass

    def audit_style(self, code: str) -> list[str]:
        """Audita el estilo básico del código (PEP 8 simplificado)."""
        issues = []
        lines = code.split("\n")

        for idx, line in enumerate(lines, 1):
            if len(line) > 79 and not ("http" in line or "https" in line):
                issues.append(
                    f"Línea {idx}: Excede los 79 caracteres recomendados por PEP 8 (longitud: {len(line)})."
                )
            if re.search(r"^[a-z]+[A-Z]+[a-zA-Z0-9]*\s*=\s*", line.strip()):
                issues.append(
                    f"Línea {idx}: Variable nombrada usando CamelCase. PEP 8 recomienda snake_case para variables (ej. nombre_variable)."
                )

        return issues

    def audit_security(self, code: str) -> list[str]:
        """Escaneo estático de seguridad (OWASP y riesgos de código generado por IA)."""
        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [
                f"Error de sintaxis crítica al parsear código para análisis de seguridad: {e}"
            ]

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ["eval", "exec"]
            ):
                issues.append(
                    f"Riesgo de Seguridad: Uso de `{node.func.id}()`. "
                    "Ejecutar cadenas de texto arbitrarias expone al sistema a inyecciones de código "
                    "y ejecución remota de comandos no autorizados (Riesgo OWASP LLM-02)."
                )

        api_key_pattern = re.compile(
            r'(api_key|token|password|secret|key|passwd)\s*=\s*[\'"][a-zA-Z0-9_\-\.]{10,}[\'"]',
            re.IGNORECASE,
        )
        for idx, line in enumerate(code.split("\n"), 1):
            if api_key_pattern.search(line) and "os.environ" not in line:
                issues.append(
                    f"Línea {idx} - Riesgo de Seguridad: Posible credencial o API Key expuesta en texto plano. "
                    "Se recomienda usar variables de entorno (ej. `os.getenv()`) y nunca commitear claves "
                    "al repositorio."
                )

        return issues

    def run_pytest(self, test_file_path: Path) -> dict[str, Any]:
        """Ejecuta pytest sobre el archivo de pruebas especificado y retorna el resultado."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file_path), "-q", "--tb=short"],
                capture_output=True,
                text=True,
                check=False,
            )
            passed = result.returncode == 0
            return {
                "passed": passed,
                "output": result.stdout if result.stdout else result.stderr,
            }
        except FileNotFoundError:
            return {
                "passed": False,
                "output": "Error: `pytest` no está instalado en el PATH o entorno actual.",
            }

    def generate_report(
        self, student_code: str, test_file_path: Path | None = None
    ) -> str:
        """Genera un reporte pedagógico completo en Markdown."""
        style_issues = self.audit_style(student_code)
        security_issues = self.audit_security(student_code)

        report = ["# [AUDITORIA] Reporte Pedagogico de Auditoria de Codigo", ""]

        report.append("## [ESTILO] Estilo y Estandares (PEP 8)")
        if style_issues:
            for issue in style_issues:
                report.append(f"- [ADVERTENCIA] {issue}")
        else:
            report.append("- [OK] El codigo cumple con las guias de estilo PEP 8 basicas.")
        report.append("")

        report.append("## [SEGURIDAD] Higiene de Seguridad (OWASP LLM & Credenciales)")
        if security_issues:
            for issue in security_issues:
                report.append(f"- [ALERTA] {issue}")
        else:
            report.append("- [OK] No se detectaron vulnerabilidades criticas comunes en el escaneo estatico.")
        report.append("")

        if test_file_path and test_file_path.exists():
            report.append("## [TESTS] Pruebas Unitarias (pytest)")
            test_results = self.run_pytest(test_file_path)
            if test_results["passed"]:
                report.append("- [OK] ¡Todas las pruebas unitarias pasaron exitosamente!")
            else:
                report.append("- [FALLO] Algunas pruebas unitarias fallaron. Detalle de la ejecucion:")
                report.append("```")
                report.append(test_results["output"])
                report.append("```")

        return "\n".join(report)


if __name__ == "__main__":
    test_code = """
def calcularTasaAprendizaje(epoca):
    API_KEY = "REEMPLAZA_CON_TU_API_KEY_1234567890"
    tasa = 0.01 / (1 + epoca)
    eval("print('Calculando...')")
    return tasa
"""
    auditor = CodeAuditorAgent()
    print("=== PROBANDO AUDITOR DE CÓDIGO ===")
    print(auditor.generate_report(test_code))
