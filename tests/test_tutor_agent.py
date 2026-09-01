"""Tests de caracterización para TutorAgent (RAG con ChromaDB + Gemini)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.multiagent_core.tutor_agent import TutorAgent


@pytest.fixture
def course_dir(tmp_path: Path) -> Path:
    lecciones_dir = tmp_path / "lecciones"
    lecciones_dir.mkdir()
    (lecciones_dir / "UNIDAD_0_PRUEBA.md").write_text(
        "# Unidad 0\n\n## Sección\n\nContenido de prueba sobre agentes de IA.",
        encoding="utf-8",
    )
    return lecciones_dir


@pytest.fixture
def tutor(course_dir: Path, tmp_path: Path) -> TutorAgent:
    return TutorAgent(
        course_dir=course_dir,
        chroma_path=tmp_path / "chroma",
        memory_path=tmp_path / "memoria.json",
        bibliografia_dir=tmp_path / "bibliografia",
    )


class TestInit:
    def test_se_instancia_sin_gemini_api_key(self, tutor: TutorAgent):
        """El agente debe poder construirse sin GEMINI_API_KEY -- solo
        ask() debe fallar de forma controlada."""
        assert tutor is not None

    def test_crea_directorio_de_chroma(self, tutor: TutorAgent, tmp_path: Path):
        assert (tmp_path / "chroma").exists()

    def test_indexa_el_contenido_de_lecciones(self, tutor: TutorAgent):
        assert tutor.collection.count() > 0


class TestAsk:
    def test_retorna_error_controlado_sin_api_key(
        self, tutor: TutorAgent, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        respuesta = tutor.ask("¿Qué es un agente de IA?")
        assert "GEMINI_API_KEY" in respuesta or "api key" in respuesta.lower()

    def test_responde_usando_gemini_mockeado(
        self, tutor: TutorAgent, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GEMINI_API_KEY", "key-de-prueba")
        mock_response = MagicMock()
        mock_response.text = "Respuesta simulada del tutor"

        with patch("src.multiagent_core.tutor_agent.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_client

            respuesta = tutor.ask("¿Qué es un agente de IA?")

        assert respuesta == "Respuesta simulada del tutor"
        mock_client_cls.assert_called_once()
