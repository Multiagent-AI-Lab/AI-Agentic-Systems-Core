"""
CONVERSOR DE MARKDOWN A NOTEBOOKS
=================================

Script de automatización que utiliza el NotebookCompilerAgent para
transformar las lecciones Markdown de `lecciones/` en Jupyter Notebooks
listos para su uso por parte de los estudiantes.

Regenerar siempre que se edite cualquier lecciones/UNIDAD_*.md — los
notebooks nunca se editan a mano.
"""

import shutil
import sys
from pathlib import Path

from src.multiagent_core.notebook_compiler_agent import NotebookCompilerAgent

if __name__ == "__main__":
    if shutil.which("npx") is None:
        print("ERROR: Node.js no está instalado o 'npx' no está en el PATH.")
        print("Los diagramas Mermaid requieren Node.js para renderizarse como SVG.")
        print("Instalar con: winget install OpenJS.NodeJS (Windows) o desde https://nodejs.org")
        sys.exit(1)

    BASE_DIR = Path(__file__).parent
    SOURCE_DIR = BASE_DIR / "lecciones"
    output_dir = BASE_DIR / "notebooks"
    output_dir.mkdir(exist_ok=True)

    files_to_convert = [
        "UNIDAD_0_FUNDAMENTOS_MATEMATICOS.md",
        "UNIDAD_1_ML_FUNDAMENTALS.md",
        "UNIDAD_2_IA_APLICADA_GENERICA.md",
        "UNIDAD_3_SISTEMAS_MULTI_AGENTE.md",
        "UNIDAD_4_SISTEMAS_ADAPTATIVOS.md",
    ]

    print("=" * 75)
    print("INICIANDO CONVERSOR DE NOTEBOOKS")
    print("=" * 75)
    print(f"Directorio Base: {BASE_DIR}")
    print(f"Directorio de Salida: {output_dir}\n")

    compiler = NotebookCompilerAgent()

    converted = 0
    missing = []

    for filename in files_to_convert:
        filepath = SOURCE_DIR / filename
        if not filepath.exists():
            missing.append(filename)
            continue

        try:
            compiler.compile(filepath, output_dir)
            converted += 1
            print(f"  [OK] {filename}")
        except (OSError, ValueError) as e:
            print(f"  [ERROR] Error al compilar {filename}: {e}")

    print("\n" + "=" * 75)
    print("RESUMEN DE EJECUCION")
    print("=" * 75)
    print(f"  [OK] Convertidos con exito: {converted}")
    if missing:
        print(
            f"  [PENDIENTE] Archivos Markdown no encontrados aun (pendientes de creacion): {len(missing)}"
        )
        for f in missing:
            print(f"  - {f}")
    print("=" * 75)

    if missing or converted < len(files_to_convert):
        sys.exit(1)
