"""
Agente Auditor de Contenido (ContentAuditorAgent)
====================================================

Audita el contenido pedagógico de las unidades del curso (contenido
escrito por el mantenedor, no por estudiantes) contra 6 dimensiones de
calidad: rigor matemático/LaTeX, patrón pedagógico central completo,
calidad de código de ejemplo, reproducibilidad de agentes, Diccionario de
Variables, e invariantes estructurales. Heurístico puro, sin LLM.
"""

import ast
import re
from pathlib import Path
from typing import Any

from ._fence_utils import extract_fenced_blocks
from .code_auditor_agent import CodeAuditorAgent
from .mermaid_renderer import MermaidRenderer

SKILL_METADATA = {
    "name": "content_auditor_agent",
    "description": "Audita el contenido pedagógico de las unidades del curso en 6 dimensiones: LaTeX, patrón pedagógico central, código de ejemplo, reproducibilidad, Diccionario de Variables, invariantes estructurales.",
    "version": "1.0.0",
    "input": "md_path: Path",
    "output": "Dict[str, Any] con hallazgos por dimensión",
    "requires_api_key": False,
}

LATEX_KNOWN_MALFORMED = {
    r"\DeltaG": r"\Delta G",
    r"\DeltaH": r"\Delta H",
    r"\DeltaS": r"\Delta S",
}

_CELDA_MAGICA_IPYTHON = re.compile(r"^\s*[%!]", re.MULTILINE)
_ARGUMENTOS_SIN_TIPO_ESPERADO = {"self", "cls"}
_ALEATORIEDAD_PATTERN = re.compile(r"\brandom\.\w+\(|\bnp\.random\.\w+\(")
_SEED_PATTERN = re.compile(r"\bseed\(")
_API_KEY_PATTERN = re.compile(
    r'(api_key|token|password|secret|key|passwd)\s*=\s*[\'"][a-zA-Z0-9_\-\.]{10,}[\'"]',
    re.IGNORECASE,
)


class ContentAuditorAgent:
    """Agente que audita el contenido pedagógico de las unidades del curso."""

    def __init__(self, mermaid_renderer: MermaidRenderer | None = None) -> None:
        self.code_auditor = CodeAuditorAgent()
        self.mermaid_renderer = mermaid_renderer or MermaidRenderer(
            output_dir=Path.cwd() / "notebooks" / "assets" / "diagramas"
        )

    def _audit_latex(self, content: str) -> list[str]:
        """Detecta delimitadores LaTeX desbalanceados y comandos mal formados."""
        hallazgos: list[str] = []

        sin_fences = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        tokens = re.findall(r"\$\$|\$", sin_fences)
        doble_count = sum(1 for t in tokens if t == "$$")
        simple_count = sum(1 for t in tokens if t == "$")
        if doble_count % 2 != 0:
            hallazgos.append(
                "Delimitadores '$$' desbalanceados: hay un número impar de "
                "bloques '$$', lo que sugiere una fórmula en bloque sin cerrar."
            )
        if simple_count % 2 != 0:
            hallazgos.append(
                "Delimitadores '$' desbalanceados: hay un número impar de '$' "
                "simples fuera de bloques '$$...$$', lo que sugiere una fórmula "
                "sin cerrar."
            )

        for malformado, correcto in LATEX_KNOWN_MALFORMED.items():
            if malformado in content:
                hallazgos.append(
                    f"Comando LaTeX mal formado: '{malformado}' encontrado; "
                    f"debería ser '{correcto}' (falta espacio, KaTeX no lo renderiza)."
                )

        return hallazgos

    def _audit_codigo(self, python_blocks: list[str]) -> list[str]:
        """Audita bloques de código Python de ejemplo (docstrings, type hints, estilo)."""
        hallazgos: list[str] = []

        for code in python_blocks:
            if _CELDA_MAGICA_IPYTHON.search(code):
                continue

            hallazgos.extend(
                h for h in self.code_auditor.audit_style(code) if "79 caracteres" not in h
            )
            hallazgos.extend(self.code_auditor.audit_security(code))

            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("test_"):
                    continue
                if ast.get_docstring(node) is None:
                    hallazgos.append(
                        f"Función '{node.name}' sin docstring en el código de ejemplo."
                    )
                args_sin_tipo = [
                    a.arg for a in node.args.args
                    if a.annotation is None and a.arg not in _ARGUMENTOS_SIN_TIPO_ESPERADO
                ]
                if args_sin_tipo or node.returns is None:
                    hallazgos.append(
                        f"Función '{node.name}' sin type hints completos "
                        f"(argumentos sin tipo: {args_sin_tipo or 'ninguno'}, "
                        f"retorno anotado: {node.returns is not None})."
                    )

        return hallazgos

    def _audit_reproducibilidad(self, python_blocks: list[str]) -> list[str]:
        """Audita reproducibilidad: seeds fijados donde hay aleatoriedad,
        y ausencia de claves de API hardcodeadas."""
        hallazgos: list[str] = []

        for code in python_blocks:
            if _ALEATORIEDAD_PATTERN.search(code) and not _SEED_PATTERN.search(code):
                hallazgos.append(
                    "Bloque de código usa aleatoriedad (random/np.random) sin "
                    "fijar una semilla (seed) — el resultado no es reproducible "
                    "entre ejecuciones."
                )
            for idx, line in enumerate(code.split("\n"), 1):
                if _API_KEY_PATTERN.search(line) and "os.environ" not in line:
                    hallazgos.append(
                        f"Línea {idx} de un bloque de ejemplo: posible clave de "
                        "API hardcodeada en texto plano."
                    )

        return hallazgos

    def _audit_pedagogico(self, bloques: list[tuple], content: str) -> list[str]:
        """Verifica que el patrón pedagógico central esté completo, y valida
        la sintaxis de cualquier diagrama Mermaid presente.

        El nombre final del patrón pedagógico central de este repo se
        decide en el Sub-proyecto 3 (contenido) — hasta entonces, los
        mensajes de hallazgo usan el marcador <PATRÓN_PEDAGÓGICO>.
        """
        hallazgos: list[str] = []

        idiomas_presentes = {lang for _, lang, _ in bloques}
        patron_esperado = {"pseudocodigo", "mermaid", "python", "pytest"}
        if not patron_esperado.issubset(idiomas_presentes):
            faltantes = patron_esperado - idiomas_presentes
            hallazgos.append(
                "Ciclo del <PATRÓN_PEDAGÓGICO> incompleto (Pseudocódigo → "
                f"Mermaid → Python → pytest): faltan bloques de tipo {sorted(faltantes)}."
            )

        mermaid_blocks = [code for _, lang, code in bloques if lang == "mermaid"]
        for diagrama in mermaid_blocks:
            try:
                self.mermaid_renderer.render(diagrama)
            except RuntimeError as e:
                if "Parse error" in str(e):
                    hallazgos.append(
                        f"Diagrama Mermaid con error de sintaxis: {str(e).splitlines()[0]}"
                    )
                else:
                    hallazgos.append(
                        "No se pudo validar el diagrama Mermaid (fallo de "
                        f"infraestructura, no de sintaxis): {str(e).splitlines()[0]}"
                    )

        return hallazgos

    def _audit_diccionario_variables(self, content: str) -> list[str]:
        """Placeholder para la Dimensión 5 (Diccionario de Variables).

        Verificar que cada símbolo del Diccionario de Variables esté usado
        en código realmente ejecutado (no solo en tabla/docstring) requiere
        el contenido pedagógico real, que llega en el Sub-proyecto 3. Esta
        tarea deja el mecanismo listo pero sin heurística activa todavía —
        no reporta hallazgos hasta que el Sub-proyecto 3 defina el formato
        exacto de la sección "### Diccionario de Variables" en este repo.
        """
        return []

    def _verifica_fences_balanceados(self, content: str) -> list[str]:
        """Detecta fences de apertura que extract_fenced_blocks() no pudo
        cerrar correctamente."""
        bloques = extract_fenced_blocks(content)
        if not bloques:
            return []

        ultimo_fence, ultimo_lang, ultimo_codigo = bloques[-1]
        if not ultimo_codigo:
            return []

        primera_linea_codigo = ultimo_codigo.split("\n")[0]
        linea_apertura = f"{ultimo_fence}{ultimo_lang}"
        marcador_apertura = f"{linea_apertura}\n{primera_linea_codigo}"
        posicion_apertura = content.rfind(marcador_apertura)
        if posicion_apertura == -1:
            return []

        resto_tras_apertura = content[posicion_apertura:]
        lineas_resto = resto_tras_apertura.split("\n")
        cierre_encontrado = any(
            linea.strip() == ultimo_fence for linea in lineas_resto[1:]
        )
        if not cierre_encontrado:
            return [
                (
                    "Posible fence de código sin cerrar: un bloque abierto con "
                    f"'{ultimo_fence}' no encuentra su línea de cierre exacta en "
                    "el resto del documento."
                )
            ]
        return []

    def _audit_estructural(self, content: str) -> list[str]:
        """Audita invariantes estructurales: fences balanceados. La
        verificación de celdas de autoevaluación consistentes se activa en
        el Sub-proyecto 3, cuando exista el formato real de autoevaluación
        de este repo."""
        return self._verifica_fences_balanceados(content)

    def audit_unit(self, md_path: Path) -> dict[str, Any]:
        """Audita una unidad del curso contra las 6 dimensiones de calidad.

        Returns:
            Diccionario con la unidad auditada, hallazgos por dimensión
            ("latex", "patron_pedagogico", "codigo", "reproducibilidad",
            "diccionario_variables", "estructural") y el total.
        """
        content = md_path.read_text(encoding="utf-8")
        bloques = extract_fenced_blocks(content)
        python_blocks = [code for _, lang, code in bloques if lang == "python"]

        hallazgos = {
            "latex": self._audit_latex(content),
            "patron_pedagogico": self._audit_pedagogico(bloques, content),
            "codigo": self._audit_codigo(python_blocks),
            "reproducibilidad": self._audit_reproducibilidad(python_blocks),
            "diccionario_variables": self._audit_diccionario_variables(content),
            "estructural": self._audit_estructural(content),
        }
        total = sum(len(v) for v in hallazgos.values())

        return {
            "unidad": md_path.name,
            "hallazgos": hallazgos,
            "total_hallazgos": total,
        }
