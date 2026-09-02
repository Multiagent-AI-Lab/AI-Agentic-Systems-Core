from src.multiagent_core.skills.memory import episodic_retriever


def test_add_and_retrieve_usa_store_path_inyectado(tmp_path):
    store_path = tmp_path / "episodic_memory.json"

    episodic_retriever.add_episode(
        content="Simulacion DFT de Au nanoclusters",
        user_id="user_test",
        metadata={"tags": ["DFT", "Au"]},
        store_path=store_path,
    )

    assert store_path.exists()
    results = episodic_retriever.retrieve(query="Au", user_id="user_test", store_path=store_path)
    assert isinstance(results, list)
    assert len(results) == 1


def test_retrieve_sin_episodios_retorna_lista_vacia(tmp_path):
    store_path = tmp_path / "episodic_memory.json"
    results = episodic_retriever.retrieve(
        query="inexistente_xyz", user_id="user_new", store_path=store_path
    )
    assert results == []


def test_dos_store_path_distintos_no_comparten_datos(tmp_path):
    store_a = tmp_path / "a.json"
    store_b = tmp_path / "b.json"

    episodic_retriever.add_episode(content="episodio en A", user_id="u", store_path=store_a)

    results_b = episodic_retriever.retrieve(query="episodio", user_id="u", store_path=store_b)
    assert results_b == []
