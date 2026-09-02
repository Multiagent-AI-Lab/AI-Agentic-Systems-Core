import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks" / "practica_u3"

EXPECTED_NOTEBOOKS = [
    "U3_01_FUNDAMENTOS_AGENTES_MODERNOS.ipynb",
    "U3_02_LANGCHAIN_AVANZADO_LANGGRAPH.ipynb",
    "U3_03_CREWAI_SISTEMAS_MULTIAGENTE.ipynb",
    "U3_04_GOOGLE_ADK_A2A_COMP.ipynb",
    "U3_05_RAG_MEMORIA_AVANZADA.ipynb",
    "U3_06_GRAPH_RAG_MEMORIA.ipynb",
    "U3_07_MULTIMODAL_PRODUCCION.ipynb",
    "U3_08_PROYECTO_INTEGRADOR.ipynb",
]

# Rutas absolutas de sistema de archivos: unidad de Windows (C:\, d:/...) o
# raíz POSIX de usuario (/home/, /Users/). No debe aparecer ninguna en el
# código fuente de los notebooks migrados.
# Dos clases de falso positivo excluidas explícitamente:
#   1. Esquemas de URL (http://, https://...): en "https://" la "s:" seguida
#      de "/" matchearía como unidad de Windows.
#   2. Secuencias de escape de string dentro del propio código fuente Python
#      (p. ej. `"Responde:\n"`, donde "e:" + "\" + "n" matchea igual). En vez
#      de excluir letras de escape tras el separador (enfoque anterior, que
#      generaba falsos negativos con rutas reales como `C:\nano\...` o
#      `C:\temp\...`, cuyo primer segmento empieza justo con una letra de
#      escape), se exige que lo que sigue al separador sea un segmento de
#      ruta plausible: 2+ caracteres de nombre de archivo/carpeta seguidos
#      de otro separador — un escape de string real (`\n`, `\t`, `"`, etc.)
#      nunca tiene esa forma.
_URL_SCHEME_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://")
_ABSOLUTE_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/][A-Za-z0-9_.-]{2,}[\\/]|/home/|/Users/")


def _has_hardcoded_absolute_path(line: str) -> bool:
    line_without_urls = _URL_SCHEME_PATTERN.sub("", line)
    return bool(_ABSOLUTE_PATH_PATTERN.search(line_without_urls))


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_notebook_existe_y_es_valido(filename):
    path = NOTEBOOKS_DIR / filename
    assert path.exists(), f"Falta {filename} en notebooks/practica_u3/"
    nb = nbformat.read(path, as_version=4)
    assert len(nb.cells) > 0


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_notebook_no_tiene_rutas_absolutas_hardcodeadas(filename):
    path = NOTEBOOKS_DIR / filename
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            assert not _has_hardcoded_absolute_path(
                line
            ), f"{filename} contiene una ruta absoluta hardcodeada: {line!r}"


@pytest.mark.parametrize("filename", EXPECTED_NOTEBOOKS)
def test_notebook_no_importa_external_skills_del_repo_de_origen(filename):
    path = NOTEBOOKS_DIR / filename
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        assert "from external_skills" not in cell.source, (
            f"{filename} todavia importa de external_skills (repo de origen) "
            "en vez de src.multiagent_core.skills"
        )


def test_convert_to_notebooks_no_lista_practica_u3():
    driver_source = (Path(__file__).parent.parent / "convert_to_notebooks.py").read_text(
        encoding="utf-8"
    )
    assert "practica_u3" not in driver_source
