"""Tests de caracterización para pdf_indexer (extracción y chunking de PDFs)."""

from src.multiagent_core.pdf_indexer import chunk_page_text, extract_pages, index_pdf


class TestChunkPageText:
    def test_retorna_lista_vacia_con_texto_vacio(self):
        assert chunk_page_text("", page_number=1) == []

    def test_retorna_un_chunk_con_texto_corto(self):
        chunks = chunk_page_text("Texto corto.", page_number=3)
        assert len(chunks) == 1
        assert chunks[0]["page"] == 3
        assert chunks[0]["text"] == "Texto corto."

    def test_respeta_max_chars(self):
        texto_largo = "Palabra. " * 500
        chunks = chunk_page_text(texto_largo, page_number=1, max_chars=200)
        assert all(len(c["text"]) <= 200 for c in chunks)

    def test_respeta_limites_de_parrafo_cuando_es_posible(self):
        texto = "Primer párrafo.\n\nSegundo párrafo."
        chunks = chunk_page_text(texto, page_number=1, max_chars=1000)
        assert len(chunks) == 1
        assert "Primer párrafo." in chunks[0]["text"]
        assert "Segundo párrafo." in chunks[0]["text"]


class TestExtractPages:
    def test_retorna_lista_vacia_con_archivo_inexistente(self, tmp_path):
        resultado = extract_pages(tmp_path / "no_existe.pdf")
        assert resultado == []


class TestIndexPdf:
    def test_retorna_lista_vacia_con_pdf_inexistente(self, tmp_path):
        resultado = index_pdf(tmp_path / "no_existe.pdf")
        assert resultado == []
