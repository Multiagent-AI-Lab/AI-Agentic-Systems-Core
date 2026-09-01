"""
Utilidad compartida de extracción de bloques de código delimitados por
fences (```) de un texto Markdown. Usada por NotebookCompilerAgent y
ContentAuditorAgent.
"""

import re


def extract_fenced_blocks(content: str) -> list[tuple[str, str, str]]:
    """Extrae bloques de código delimitados por fences ``` de un texto Markdown.

    Respeta fences anidados: un bloque exterior de 4+ backticks que contiene
    fences de 3 backticks en su interior se extrae completo, sin cortarse en
    el primer fence interno (comportamiento CommonMark estándar).

    Args:
        content: Texto Markdown completo a escanear.

    Returns:
        Lista de tuplas (fence, language, code_content) en orden de aparición.
        `fence` es la secuencia de backticks original (p. ej. "```" o "````").
        `language` es el identificador de lenguaje tras el fence (puede ser "").
        `code_content` es el texto entre el fence de apertura y cierre.
    """
    lines = content.split("\n")
    blocks: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        fence_match = re.match(r"^(`{3,})(.*)$", line)
        if fence_match:
            fence = fence_match.group(1)
            language = fence_match.group(2).strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != fence:
                code_lines.append(lines[i])
                i += 1
            blocks.append((fence, language, "\n".join(code_lines)))
        i += 1
    return blocks
