"""src/multiagent_core/skills/routing/task_classifier.py
Clasifica tareas y las enruta al agente apropiado.
"""

from __future__ import annotations

SKILL_METADATA = {
    "name": "task_classifier",
    "domain": "routing",
    "description": "Clasifica tareas en categorías y recomienda el agente óptimo usando reglas y LLM.",
    "version": "1.0.0",
    "input": "task: str, available_agents: list[str] | None",
    "output": "dict con category, recommended_agent, confidence, reasoning, all_scores",
    "dependencies": [],
}

# Reglas de routing por dominio: keyword → (category, agent, base_confidence)
ROUTING_RULES: dict[str, list[tuple[str, str, float]]] = {
    "code_generation": [
        ("genera", "code_generation", 0.8),
        ("implementa", "code_generation", 0.8),
        ("escribe el código", "code_generation", 0.9),
        ("función", "code_generation", 0.7),
        ("script", "code_generation", 0.75),
        ("programa", "code_generation", 0.7),
        ("código para", "code_generation", 0.85),
    ],
    "data_analysis": [
        ("analiza", "data_analysis", 0.7),
        ("dataset", "data_analysis", 0.8),
        ("datos", "data_analysis", 0.65),
        ("estadística", "data_analysis", 0.8),
        ("visualiza", "data_analysis", 0.75),
        ("gráfica", "data_analysis", 0.7),
        ("csv", "data_analysis", 0.85),
    ],
    "research": [
        ("busca", "research", 0.7),
        ("investiga", "research", 0.8),
        ("papers", "research", 0.85),
        ("literatura", "research", 0.85),
        ("artículos", "research", 0.8),
        ("estado del arte", "research", 0.9),
        ("references", "research", 0.75),
    ],
    "writing": [
        ("escribe", "writing", 0.7),
        ("redacta", "writing", 0.8),
        ("reporte", "writing", 0.75),
        ("resumen", "writing", 0.7),
        ("documenta", "writing", 0.75),
        ("informe", "writing", 0.75),
    ],
    "question_answering": [
        ("qué es", "question_answering", 0.8),
        ("cómo funciona", "question_answering", 0.8),
        ("explica", "question_answering", 0.75),
        ("define", "question_answering", 0.8),
        ("cuál es", "question_answering", 0.7),
    ],
}

# Mapeo default category → recommended_agent
_DEFAULT_AGENTS: dict[str, str] = {
    "code_generation": "code_agent",
    "data_analysis": "data_agent",
    "research": "research_agent",
    "writing": "writer_agent",
    "question_answering": "qa_agent",
    "unknown": "general_agent",
}


def classify(task: str, available_agents: list[str] | None = None) -> dict:
    """
    Clasifica una tarea usando reglas de palabras clave.

    Args:
        task: Descripción de la tarea en lenguaje natural.
        available_agents: Lista de agentes disponibles. Si se provee, el
                          recommend_agent estará restringido a esta lista.

    Returns:
        dict con:
            - category (str): Categoría de la tarea.
            - recommended_agent (str): Agente recomendado.
            - confidence (float): Confianza 0.0-1.0.
            - reasoning (str): Explicación de la decisión.
            - all_scores (dict): Scores para todas las categorías.
    """
    task_lower = task.lower()
    all_scores: dict[str, float] = {cat: 0.0 for cat in ROUTING_RULES}

    # Calcular score por categoría según coincidencias de keywords
    for category, rules in ROUTING_RULES.items():
        for keyword, _, base_conf in rules:
            if keyword in task_lower:
                # Acumular: múltiples keywords de la misma categoría refuerzan la confianza
                all_scores[category] = min(1.0, all_scores[category] + base_conf * 0.5)

    # Limitar cada score a 1.0
    all_scores = {k: min(1.0, v) for k, v in all_scores.items()}

    best_category = max(all_scores, key=lambda k: all_scores[k])
    best_score = all_scores[best_category]

    # Umbral mínimo de confianza para no devolver falso positivo
    if best_score < 0.15:
        best_category = "unknown"
        best_score = 0.05
        reasoning = "No se encontraron palabras clave suficientes para clasificar la tarea."
    else:
        matching = [kw for kw, _, _ in ROUTING_RULES.get(best_category, []) if kw in task_lower]
        reasoning = f"Categoría '{best_category}' activada por: {', '.join(matching[:3])}."

    # Seleccionar agente
    default_agent = _DEFAULT_AGENTS.get(best_category, "general_agent")
    if available_agents:
        # Usar el primer agente disponible que coincida con la categoría
        preferred = [a for a in available_agents if best_category.split("_")[0] in a.lower()]
        recommended_agent = preferred[0] if preferred else available_agents[0]
    else:
        recommended_agent = default_agent

    return {
        "category": best_category,
        "recommended_agent": recommended_agent,
        "confidence": round(best_score, 3),
        "reasoning": reasoning,
        "all_scores": {k: round(v, 3) for k, v in all_scores.items()},
    }


# Alias para compatibilidad con U5_06
classify_task = classify
