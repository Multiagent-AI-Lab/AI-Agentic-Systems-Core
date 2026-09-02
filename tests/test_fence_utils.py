"""Tests para la utilidad compartida de extracción de bloques con fences."""

from src.multiagent_core._fence_utils import _scan_fence_segments, extract_fenced_blocks


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


class TestScanFenceSegments:
    """`_scan_fence_segments` es el escaneo de bajo nivel compartido: tanto
    `extract_fenced_blocks` (descarta el texto circundante) como
    `NotebookCompilerAgent._segment_document` (lo conserva, para preservar
    el orden real del documento) derivan su resultado de esta única
    implementación del reconocimiento de fences — eliminando la
    duplicación de lógica que existía entre los dos módulos."""

    def test_segmenta_texto_y_codigo_en_orden(self):
        md = "texto1\n\n```python\nx = 1\n```\n\ntexto2"
        segmentos = _scan_fence_segments(md)
        assert segmentos[0] == ("text", "texto1\n")
        assert segmentos[1] == ("code", "```", "python", "x = 1")
        assert segmentos[2] == ("text", "\ntexto2")

    def test_respeta_fences_anidados(self):
        md = "````markdown\n```python\nx = 1\n```\n````"
        segmentos = _scan_fence_segments(md)
        bloques_codigo = [s for s in segmentos if s[0] == "code"]
        assert len(bloques_codigo) == 1
        assert bloques_codigo[0][1] == "````"
        assert "```python" in bloques_codigo[0][3]

    def test_solo_texto_sin_fences(self):
        segmentos = _scan_fence_segments("solo texto plano")
        assert segmentos == [("text", "solo texto plano")]

    def test_extract_fenced_blocks_es_consistente_con_scan_fence_segments(self):
        """extract_fenced_blocks debe retornar exactamente los bloques
        'code' de _scan_fence_segments, sin el texto circundante — mismo
        escaneo de fences, dos vistas distintas del resultado."""
        md = "texto1\n\n```python\nx = 1\n```\n\ntexto2\n\n```js\ny = 2\n```"
        segmentos = _scan_fence_segments(md)
        esperado = [(s[1], s[2], s[3]) for s in segmentos if s[0] == "code"]
        assert extract_fenced_blocks(md) == esperado
