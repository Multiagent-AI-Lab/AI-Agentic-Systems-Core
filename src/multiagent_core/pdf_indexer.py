"""
pdf_indexer: extracción de texto y chunking de PDFs para indexar la
bibliografía académica del curso (bibliografia/*.pdf) en el RAG de
TutorAgent, en una colección ChromaDB separada de las lecciones.
"""

import logging
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 1000

_SURROGATE_HUERFANO_RE = re.compile("[\ud800-\udfff]")


def extract_pages(pdf_path: Path) -> list[str]:
    """Extrae el texto de cada página de un PDF, en orden."""
    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError) as e:
        logger.error("No se pudo abrir %s como PDF: %s", pdf_path.name, e)
        return []

    paginas = []
    for numero_pagina, page in enumerate(reader.pages, start=1):
        try:
            texto = page.extract_text() or ""
        except (PyPdfError, ValueError) as e:
            logger.warning(
                "No se pudo extraer texto de la página %d de %s: %s",
                numero_pagina, pdf_path.name, e,
            )
            texto = ""
        texto, n_surrogates = _SURROGATE_HUERFANO_RE.subn("", texto)
        if n_surrogates:
            logger.warning(
                "Página %d de %s contenía %d carácter(es) surrogate UTF-16 huérfano(s); se eliminaron.",
                numero_pagina, pdf_path.name, n_surrogates,
            )
        paginas.append(texto)

    return paginas


def chunk_page_text(text: str, page_number: int, max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """Trocea el texto de una página en bloques de como máximo `max_chars` caracteres."""
    if not text.strip():
        return []

    parrafos = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parrafos:
        parrafos = [text.strip()]

    unidades: list[str] = []
    for parrafo in parrafos:
        if len(parrafo) <= max_chars:
            unidades.append(parrafo)
        else:
            unidades.extend(_dividir_texto_largo(parrafo, max_chars))

    chunks: list[dict] = []
    bloque_actual = ""

    for unidad in unidades:
        candidato = f"{bloque_actual}\n\n{unidad}" if bloque_actual else unidad
        if len(candidato) > max_chars and bloque_actual:
            chunks.append({"text": bloque_actual, "page": page_number})
            bloque_actual = unidad
        else:
            bloque_actual = candidato

    if bloque_actual:
        chunks.append({"text": bloque_actual, "page": page_number})

    return chunks


def _dividir_texto_largo(texto: str, max_chars: int) -> list[str]:
    """Subdivide un bloque de texto que excede max_chars, respetando oraciones."""
    oraciones = [o.strip() for o in texto.split(". ") if o.strip()]

    partes: list[str] = []
    bloque_actual = ""

    for i, oracion in enumerate(oraciones):
        oracion_con_punto = (
            oracion if oracion.endswith(".") or i == len(oraciones) - 1 else f"{oracion}."
        )
        if len(oracion_con_punto) > max_chars:
            if bloque_actual:
                partes.append(bloque_actual)
                bloque_actual = ""
            partes.extend(
                oracion_con_punto[j : j + max_chars]
                for j in range(0, len(oracion_con_punto), max_chars)
            )
            continue

        candidato = f"{bloque_actual} {oracion_con_punto}" if bloque_actual else oracion_con_punto
        if len(candidato) > max_chars and bloque_actual:
            partes.append(bloque_actual)
            bloque_actual = oracion_con_punto
        else:
            bloque_actual = candidato

    if bloque_actual:
        partes.append(bloque_actual)

    return partes


def index_pdf(pdf_path: Path, max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """Extrae y trocea un PDF completo."""
    paginas = extract_pages(pdf_path)

    chunks: list[dict] = []
    for numero_pagina, texto_pagina in enumerate(paginas, start=1):
        for chunk in chunk_page_text(texto_pagina, numero_pagina, max_chars):
            chunks.append({**chunk, "source": pdf_path.name})

    return chunks
