"""Tests para la utilidad compartida de extracción de bloques con fences."""

from src.multiagent_core._fence_utils import extract_fenced_blocks


class TestExtractFencedBlocks:
    def test_extrae_un_bloque_simple(self):
        md = "texto\n```python\nx = 1\n```\nmás texto"
        bloques = extract_fenced_blocks(md)
        assert bloques == [("```", "python", "x = 1")]

    def test_respeta_fences_anidados(self):
        md = "````markdown\n```python\nx = 1\n```\n````"
        bloques = extract_fenced_blocks(md)
        assert len(bloques) == 1
        assert bloques[0][0] == "````"
        assert "```python" in bloques[0][2]

    def test_retorna_lista_vacia_sin_fences(self):
        assert extract_fenced_blocks("solo texto plano") == []
