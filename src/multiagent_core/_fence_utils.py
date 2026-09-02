"""
Utilidad compartida de extracción de bloques de código delimitados por
fences (```) de un texto Markdown. Usada por NotebookCompilerAgent y
ContentAuditorAgent.
"""

import re

_FENCE_OPEN_RE = re.compile(r"^(`{3,})(.*)$")


def _scan_fence_segments(content: str) -> list[tuple]:
    """Escaneo de bajo nivel compartido: recorre el documento una sola vez
    y produce segmentos ordenados según aparecen en el texto fuente.

    Única implementación real del reconocimiento de fences del repo — tanto
    `extract_fenced_blocks` (descarta el texto circundante) como
    `NotebookCompilerAgent._segment_document` (lo conserva, para preservar
    el orden real del documento al compilar un notebook) derivan su
    resultado de esta función, en vez de reimplementar el mismo bucle.

    Respeta fences anidados: un bloque exterior de 4+ backticks que contiene
    fences de 3 backticks en su interior se extrae completo, sin cortarse en
    el primer fence interno (comportamiento CommonMark estándar).

    Args:
        content: Texto Markdown completo a escanear.

    Returns:
        Lista de tuplas en orden de aparición:
        - `("text", texto)` para un tramo de Markdown entre fences.
        - `("code", fence, language, code_content)` para un bloque
          delimitado por fences. `fence` es la secuencia de backticks
          original (p. ej. "```" o "````"); `language` es el identificador
          de lenguaje tras el fence (puede ser ""); `code_content` es el
          texto entre el fence de apertura y cierre.
    """
    lines = content.split("\n")
    segmentos: list[tuple] = []
    texto_actual: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        fence_match = _FENCE_OPEN_RE.match(line)
        if fence_match:
            if texto_actual:
                segmentos.append(("text", "\n".join(texto_actual)))
                texto_actual = []

            fence = fence_match.group(1)
            language = fence_match.group(2).strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != fence:
                code_lines.append(lines[i])
                i += 1
            segmentos.append(("code", fence, language, "\n".join(code_lines)))
        else:
            texto_actual.append(line)
        i += 1

    if texto_actual:
        segmentos.append(("text", "\n".join(texto_actual)))

    return segmentos


def extract_fenced_blocks(content: str) -> list[tuple[str, str, str]]:
    """Extrae bloques de código delimitados por fences ``` de un texto Markdown.

    Vista sobre `_scan_fence_segments`: retorna solo los segmentos de
    código, descartando el texto circundante.

    Args:
        content: Texto Markdown completo a escanear.

    Returns:
        Lista de tuplas (fence, language, code_content) en orden de aparición.
        `fence` es la secuencia de backticks original (p. ej. "```" o "````").
        `language` es el identificador de lenguaje tras el fence (puede ser "").
        `code_content` es el texto entre el fence de apertura y cierre.
    """
    return [
        segmento[1:]
        for segmento in _scan_fence_segments(content)
        if segmento[0] == "code"
    ]
