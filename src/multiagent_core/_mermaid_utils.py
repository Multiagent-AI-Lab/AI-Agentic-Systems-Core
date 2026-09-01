"""
Utilidad compartida de generación de IDs de nodo Mermaid.

Usada por FlowchartAgent y PseudocodeAgent, que generan diagramas de flujo
en formato Mermaid con la misma convención de identificadores (node_1,
node_2, ...).
"""


class MermaidNodeCounter:
    """Genera identificadores secuenciales de nodo para diagramas Mermaid."""

    def __init__(self) -> None:
        self._count = 0

    def next_id(self) -> str:
        """Genera el siguiente identificador de nodo.

        Returns:
            Identificador con el formato "node_N", donde N inicia en 1.
        """
        self._count += 1
        return f"node_{self._count}"

    def reset(self) -> None:
        """Reinicia el contador a cero."""
        self._count = 0
