"""
Agente Generador de Diagramas (FlowchartAgent)
===============================================

Genera de forma automática un diagrama de flujo en formato Mermaid a partir
de una función en Python usando el árbol de sintaxis abstracta (AST).
"""

import ast

from ._mermaid_utils import MermaidNodeCounter

SKILL_METADATA = {
    "name": "flowchart_agent",
    "description": "Genera diagramas de flujo Mermaid a partir del AST de una función Python.",
    "version": "1.0.0",
    "input": "code_source: str",
    "output": "str (diagrama Mermaid en formato graph TD)",
    "requires_api_key": False,
}


class FlowchartAgent:
    """Agente que traduce código Python a sintaxis de diagramas de flujo Mermaid."""

    def __init__(self) -> None:
        self._node_counter = MermaidNodeCounter()

    def _next_node_id(self) -> str:
        return self._node_counter.next_id()

    def build_mermaid_flowchart(self, code_source: str) -> str:
        """Lee un código fuente en Python y genera la cadena del diagrama Mermaid."""
        try:
            tree = ast.parse(code_source)
        # Código de entrada puede fallar de formas no previstas por
        # SyntaxError; nunca debe tumbar el pipeline de auditoría.
        except Exception as e:  # noqa: BLE001
            return f"%% Error al parsear código: {e}\n"

        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_node = node
                break

        if not func_node:
            return "%% No se encontró una definición de función (def) para diagramar.\n"

        self._node_counter.reset()
        mermaid_lines = ["graph TD", f"    start([Inicio: {func_node.name}])"]

        last_node = "start"
        mermaid_lines, _ = self._process_body(func_node.body, mermaid_lines, last_node)

        return "\n".join(mermaid_lines)

    def _process_body(
        self, statements, lines: list, last_node: str
    ) -> tuple[list, str]:
        """Procesa una secuencia de sentencias (body) recursivamente."""
        current_last = last_node

        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                node_id = self._next_node_id()
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                targets_str = ", ".join(targets) if targets else "var"
                lines.append(f'    {node_id}["Asignar: {targets_str}"]')
                lines.append(f"    {current_last} --> {node_id}")
                current_last = node_id

            elif isinstance(stmt, ast.If):
                cond_id = self._next_node_id()
                cond_text = "Condición"
                lines.append(f'    {cond_id}{{{{"{cond_text}?"}}}}')
                lines.append(f"    {current_last} --> {cond_id}")

                true_id = self._next_node_id()
                lines.append(f'    {true_id}["Rama Verdadero (Si)"]')
                lines.append(f"    {cond_id} -- Sí --> {true_id}")
                lines, last_true = self._process_body(stmt.body, lines, true_id)

                if stmt.orelse:
                    false_id = self._next_node_id()
                    lines.append(f'    {false_id}["Rama Falso (Sino)"]')
                    lines.append(f"    {cond_id} -- No --> {false_id}")
                    lines, last_false = self._process_body(stmt.orelse, lines, false_id)

                    join_id = self._next_node_id()
                    lines.append(f'    {join_id}["Unión Condicional"]')
                    lines.append(f"    {last_true} --> {join_id}")
                    lines.append(f"    {last_false} --> {join_id}")
                    current_last = join_id
                else:
                    join_id = self._next_node_id()
                    lines.append(f'    {join_id}["Unión Condicional"]')
                    lines.append(f"    {cond_id} -- No --> {join_id}")
                    lines.append(f"    {last_true} --> {join_id}")
                    current_last = join_id

            elif isinstance(stmt, (ast.While, ast.For)):
                loop_cond_id = self._next_node_id()
                loop_type = "Mientras" if isinstance(stmt, ast.While) else "Para"
                lines.append(f'    {loop_cond_id}{{{{"{loop_type} Iterar?"}}}}')
                lines.append(f"    {current_last} --> {loop_cond_id}")

                body_start_id = self._next_node_id()
                lines.append(f'    {body_start_id}["Cuerpo del Bucle"]')
                lines.append(f"    {loop_cond_id} -- Sí --> {body_start_id}")

                lines, last_body_node = self._process_body(
                    stmt.body, lines, body_start_id
                )
                lines.append(f"    {last_body_node} --> {loop_cond_id}")

                exit_id = self._next_node_id()
                lines.append(f'    {exit_id}["Salir Bucle"]')
                lines.append(f"    {loop_cond_id} -- No --> {exit_id}")
                current_last = exit_id

            elif isinstance(stmt, ast.Return):
                node_id = self._next_node_id()
                lines.append(f"    {node_id}([Fin / Retorno])")
                lines.append(f"    {current_last} --> {node_id}")
                current_last = node_id

        return lines, current_last


if __name__ == "__main__":
    example_code = """
def calcular_tasa_aprendizaje(epoca, lr_inicial):
    if epoca > 10:
        lr = lr_inicial * 0.1
    else:
        lr = lr_inicial
    return lr
"""
    agent = FlowchartAgent()
    print("=== PROBANDO GENERADOR DE MERMAID ===")
    print(agent.build_mermaid_flowchart(example_code))
