"""Tests de caracterización para FlowchartAgent (AST de Python -> Mermaid)."""

from src.multiagent_core.flowchart_agent import FlowchartAgent


class TestBuildMermaidFlowchart:
    def test_genera_diagrama_con_nodo_de_inicio(self):
        agent = FlowchartAgent()
        codigo = "def calcular_tasa(lr):\n    return lr * 0.9\n"
        diagrama = agent.build_mermaid_flowchart(codigo)
        assert "graph TD" in diagrama
        assert "Inicio: calcular_tasa" in diagrama

    def test_genera_nodo_de_asignacion(self):
        agent = FlowchartAgent()
        codigo = "def f(x):\n    y = x * 2\n    return y\n"
        diagrama = agent.build_mermaid_flowchart(codigo)
        assert "Asignar: y" in diagrama

    def test_genera_nodo_condicional_con_ambas_ramas(self):
        agent = FlowchartAgent()
        codigo = (
            "def f(x):\n"
            "    if x > 0:\n"
            "        y = 1\n"
            "    else:\n"
            "        y = -1\n"
            "    return y\n"
        )
        diagrama = agent.build_mermaid_flowchart(codigo)
        assert "Rama Verdadero (Si)" in diagrama
        assert "Rama Falso (Sino)" in diagrama

    def test_genera_nodo_de_ciclo_for(self):
        agent = FlowchartAgent()
        codigo = "def f(items):\n    for x in items:\n        y = x\n    return y\n"
        diagrama = agent.build_mermaid_flowchart(codigo)
        assert "Para Iterar?" in diagrama

    def test_genera_nodo_de_ciclo_while(self):
        agent = FlowchartAgent()
        codigo = "def f(n):\n    while n > 0:\n        n = n - 1\n    return n\n"
        diagrama = agent.build_mermaid_flowchart(codigo)
        assert "Mientras Iterar?" in diagrama

    def test_retorna_comentario_si_no_hay_funcion(self):
        agent = FlowchartAgent()
        diagrama = agent.build_mermaid_flowchart("x = 1\n")
        assert diagrama.startswith("%%")
        assert "no se encontró" in diagrama.lower()

    def test_retorna_comentario_con_error_de_sintaxis(self):
        agent = FlowchartAgent()
        diagrama = agent.build_mermaid_flowchart("def f(:\n    pass")
        assert diagrama.startswith("%%")

    def test_ids_de_nodo_se_reinician_entre_llamadas(self):
        """Regresión: dos llamadas sucesivas deben producir el mismo primer ID
        (node_1 para la primera asignación), no continuar el contador."""
        agent = FlowchartAgent()
        codigo = "def f(x):\n    y = x\n    return y\n"
        primera = agent.build_mermaid_flowchart(codigo)
        segunda = agent.build_mermaid_flowchart(codigo)
        assert primera == segunda
