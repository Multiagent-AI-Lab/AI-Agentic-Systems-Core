"""Tests TDD para la utilidad compartida de generación de IDs de nodo Mermaid."""

from src.multiagent_core._mermaid_utils import MermaidNodeCounter


class TestMermaidNodeCounter:
    def test_primer_id_es_node_1(self):
        counter = MermaidNodeCounter()
        assert counter.next_id() == "node_1"

    def test_ids_incrementan_secuencialmente(self):
        counter = MermaidNodeCounter()
        assert counter.next_id() == "node_1"
        assert counter.next_id() == "node_2"
        assert counter.next_id() == "node_3"

    def test_reset_reinicia_el_contador(self):
        counter = MermaidNodeCounter()
        counter.next_id()
        counter.next_id()
        counter.reset()
        assert counter.next_id() == "node_1"
