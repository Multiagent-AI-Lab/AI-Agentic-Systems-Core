"""Tests para el patrón de detección de credenciales compartido entre agentes."""

from src.multiagent_core._security_patterns import API_KEY_PATTERN


class TestApiKeyPattern:
    def test_detecta_asignacion_de_api_key_con_comillas(self):
        assert API_KEY_PATTERN.search('API_KEY = "sk-1234567890abcdef"')

    def test_detecta_variantes_de_nombre(self):
        assert API_KEY_PATTERN.search('token = "abcdefghij1234567890"')
        assert API_KEY_PATTERN.search('password = "abcdefghij1234567890"')
        assert API_KEY_PATTERN.search('secret = "abcdefghij1234567890"')
        assert API_KEY_PATTERN.search('passwd = "abcdefghij1234567890"')

    def test_no_detecta_valor_corto(self):
        assert not API_KEY_PATTERN.search('key = "corto"')

    def test_no_detecta_sin_comillas(self):
        assert not API_KEY_PATTERN.search("api_key = sk1234567890abcdef")
