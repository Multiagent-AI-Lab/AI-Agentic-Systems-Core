"""
Agente Tutor (TutorAgent)
============================

Responde preguntas del curso vía RAG semántico (ChromaDB) sobre el
contenido de lecciones/ y bibliografía académica en PDF, usando Gemini
para generar la respuesta final. El Client de Gemini se crea de forma
perezosa dentro de ask() -- el agente es instanciable sin GEMINI_API_KEY.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from google import genai

from .pdf_indexer import index_pdf

SKILL_METADATA = {
    "name": "tutor_agent",
    "description": "Responde dudas del curso vía RAG semántico (ChromaDB) + Gemini.",
    "version": "1.0.0",
    "input": "question: str (ask) | course_dir: Path, chroma_path, memory_path, bibliografia_dir (constructor)",
    "output": "str (respuesta en Markdown)",
    "requires_api_key": True,
}

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHROMA_DIRNAME = ".chroma"
DEFAULT_MEMORY_FILENAME = ".tutor_memory.json"


class TutorAgent:
    """Agente que responde preguntas del curso vía RAG + Gemini."""

    def __init__(
        self,
        course_dir: Path,
        chroma_path: Path | None = None,
        memory_path: Path | None = None,
        bibliografia_dir: Path | None = None,
    ) -> None:
        load_dotenv()
        self.course_dir = Path(course_dir)
        self.model_name = "gemini-2.5-flash"
        self.chroma_path = (
            Path(chroma_path) if chroma_path else self.course_dir / DEFAULT_CHROMA_DIRNAME
        )
        self.memory_path = (
            Path(memory_path) if memory_path else self.course_dir / DEFAULT_MEMORY_FILENAME
        )
        self.bibliografia_dir = (
            Path(bibliografia_dir) if bibliografia_dir else self.course_dir.parent / "bibliografia"
        )
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        self.collection = self._get_or_create_collection()
        self.bibliografia_collection = self._get_or_create_bibliografia_collection()
        self._build_index()
        self._build_bibliografia_index()

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self.chroma_client.get_or_create_collection(
            name="lecciones", embedding_function=self.embedding_fn
        )

    def _get_or_create_bibliografia_collection(self) -> chromadb.Collection:
        return self.chroma_client.get_or_create_collection(
            name="bibliografia", embedding_function=self.embedding_fn
        )

    def _get_markdown_files(self) -> list[Path]:
        return sorted(self.course_dir.glob("*.md"))

    def _split_into_sections(self, content: str) -> list[str]:
        secciones = [s.strip() for s in content.split("\n## ") if s.strip()]
        return secciones if secciones else [content]

    def _build_index(self) -> None:
        if self.collection.count() > 0:
            return

        documentos, ids, metadatas = [], [], []
        for md_file in self._get_markdown_files():
            content = md_file.read_text(encoding="utf-8")
            for idx, seccion in enumerate(self._split_into_sections(content)):
                documentos.append(seccion)
                ids.append(f"{md_file.stem}_{idx}")
                metadatas.append({"source": md_file.name})

        if documentos:
            self.collection.add(documents=documentos, ids=ids, metadatas=metadatas)

    def _build_bibliografia_index(self) -> None:
        if self.bibliografia_collection.count() > 0:
            return
        if not self.bibliografia_dir.exists():
            return

        documentos, ids, metadatas = [], [], []
        for pdf_path in sorted(self.bibliografia_dir.glob("*.pdf")):
            for idx, chunk in enumerate(index_pdf(pdf_path)):
                documentos.append(chunk["text"])
                ids.append(f"{pdf_path.stem}_{idx}")
                metadatas.append({"source": chunk["source"], "page": chunk["page"]})

        if documentos:
            self.bibliografia_collection.add(documents=documentos, ids=ids, metadatas=metadatas)

    def _search_local_docs(self, query: str) -> str:
        resultados = self.collection.query(query_texts=[query], n_results=3)
        docs = resultados.get("documents", [[]])[0]
        if not docs:
            return "Sin contexto local relevante."
        return "\n\n---\n\n".join(docs)

    def ask(self, question: str) -> str:
        """Responde una pregunta usando RAG local + Gemini.

        El Client de Gemini se crea aquí (perezoso), no en __init__, para
        que el agente sea instanciable sin GEMINI_API_KEY -- solo este
        método retorna un mensaje de error controlado si falta.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return (
                "No se puede responder: falta la variable de entorno "
                "GEMINI_API_KEY. Contexto local relevante:\n\n"
                + self._search_local_docs(question)
            )

        contexto = self._search_local_docs(question)
        client = genai.Client(api_key=api_key)

        prompt = (
            f"Contexto del curso:\n{contexto}\n\n"
            f"Pregunta del estudiante: {question}\n\n"
            "Responde de forma clara y pedagógica, citando el contexto cuando aplique."
        )
        response = client.models.generate_content(model=self.model_name, contents=prompt)
        return response.text
