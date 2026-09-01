"""Tests de caracterización para PseudocodeAgent."""

from src.multiagent_core.pseudocode_agent import PseudocodeAgent


class TestPseudocodeToMermaid:
    def test_genera_diagrama_con_inicio_y_fin(self):
        agent = PseudocodeAgent()
        diagrama = agent.pseudocode_to_mermaid("ESCRIBIR resultado")
        assert "Inicio" in diagrama
        assert "Fin" in diagrama

    def test_traduce_si_sino_fin_si(self):
        agent = PseudocodeAgent()
        pseudocodigo = "SI x > 0 ENTONCES\n    y <- 1\nSINO\n    y <- -1\nFIN_SI"
        diagrama = agent.pseudocode_to_mermaid(pseudocodigo)
        assert "SI x > 0" in diagrama
        assert "Rama SINO" in diagrama

    def test_traduce_para(self):
        agent = PseudocodeAgent()
        diagrama = agent.pseudocode_to_mermaid("PARA i EN rango HACER\n    ESCRIBIR i\nFIN_PARA")
        assert "PARA i EN rango HACER" in diagrama

    def test_traduce_mientras(self):
        agent = PseudocodeAgent()
        diagrama = agent.pseudocode_to_mermaid("MIENTRAS n > 0 HACER\n    n <- n - 1\nFIN_MIENTRAS")
        assert "MIENTRAS n > 0 HACER" in diagrama


class TestPythonToPseudocode:
    def test_traduce_funcion_con_asignacion_y_retorno(self):
        agent = PseudocodeAgent()
        codigo = "def f(x):\n    y = x * 2\n    return y\n"
        pseudocodigo = agent.python_to_pseudocode(codigo)
        assert "FUNCIÓN f(x)" in pseudocodigo
        assert "y <- x * 2" in pseudocodigo
        assert "RETORNAR y" in pseudocodigo
        assert "FIN_FUNCIÓN" in pseudocodigo

    def test_traduce_if_else(self):
        agent = PseudocodeAgent()
        codigo = "def f(x):\n    if x > 0:\n        y = 1\n    else:\n        y = -1\n    return y\n"
        pseudocodigo = agent.python_to_pseudocode(codigo)
        assert "SI x > 0 ENTONCES" in pseudocodigo
        assert "SINO" in pseudocodigo
        assert "FIN_SI" in pseudocodigo

    def test_traduce_for(self):
        agent = PseudocodeAgent()
        codigo = "def f(items):\n    for x in items:\n        y = x\n    return y\n"
        pseudocodigo = agent.python_to_pseudocode(codigo)
        assert "PARA x EN items HACER" in pseudocodigo
        assert "FIN_PARA" in pseudocodigo

    def test_traduce_while(self):
        agent = PseudocodeAgent()
        codigo = "def f(n):\n    while n > 0:\n        n = n - 1\n    return n\n"
        pseudocodigo = agent.python_to_pseudocode(codigo)
        assert "MIENTRAS n > 0 HACER" in pseudocodigo
        assert "FIN_MIENTRAS" in pseudocodigo

    def test_traduce_print_a_escribir(self):
        agent = PseudocodeAgent()
        codigo = "def f(x):\n    print(x)\n"
        pseudocodigo = agent.python_to_pseudocode(codigo)
        assert "ESCRIBIR x" in pseudocodigo

    def test_retorna_comentario_si_no_hay_funcion(self):
        agent = PseudocodeAgent()
        pseudocodigo = agent.python_to_pseudocode("x = 1\n")
        assert pseudocodigo.startswith("#")


class TestPseudocodeToPythonSkeleton:
    def test_genera_firma_y_docstring(self):
        agent = PseudocodeAgent()
        esqueleto = agent.pseudocode_to_python_skeleton(
            "FUNCIÓN calcular_promedio(valores)\n    RETORNAR promedio\nFIN_FUNCIÓN"
        )
        assert "def calcular_promedio(valores: float) -> float:" in esqueleto
        assert '"""' in esqueleto
        assert "pass" in esqueleto

    def test_retorna_comentario_si_no_hay_bloque_funcion(self):
        agent = PseudocodeAgent()
        esqueleto = agent.pseudocode_to_python_skeleton("ESCRIBIR resultado")
        assert esqueleto.startswith("#")
