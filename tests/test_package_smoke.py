"""Test de humo: confirma que el paquete src.multiagent_core es importable
y que pytest descubre correctamente los tests bajo tests/.

Este test no verifica ningún agente — el Sub-proyecto 2 agrega el primer
agente real con su propio test TDD. Su único propósito es detectar de
inmediato un problema de packaging/sys.path antes de que el Sub-proyecto 2
empiece a escribir código sobre una base rota.
"""

import src.multiagent_core


def test_multiagent_core_package_is_importable():
    assert src.multiagent_core is not None
