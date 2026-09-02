"""src/multiagent_core/skills/evaluation/output_scorer.py
Evalúa la calidad de outputs de agentes LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SKILL_METADATA = {
    "name": "output_scorer",
    "domain": "evaluation",
    "description": "Evalúa outputs de LLMs con scoring heurístico y opcionalmente con un LLM juez.",
    "version": "1.1.0",
    "input": "output: str, task_description: str, criteria: dict[str, float] | None, llm",
    "output": "ScoreResult con score, max_score, breakdown, feedback",
    "dependencies": [],
}

# Criterios por defecto — pesos suman exactamente 1.0
DEFAULT_CRITERIA: dict[str, float] = {
    "length": 0.20,
    "structure": 0.25,
    "specificity": 0.30,
    "actionability": 0.25,
}


@dataclass
class ScoreResult:
    """Resultado de la evaluación de un output de agente."""

    score: float  # Score ponderado total (0.0 a max_score)
    max_score: float  # Score máximo posible (1.0)
    breakdown: dict  # Score por dimensión (0.0 a 1.0)
    feedback: str  # Texto de feedback consolidado
    passed: bool = True  # True si score/max_score >= passing_threshold


def _score_dimension_heuristic(name: str, text: str) -> float:
    """Retorna score 0.0–1.0 para una dimensión dada."""
    word_count = len(text.split())

    if name == "length":
        if word_count < 10:
            return 0.05
        elif word_count < 30:
            return 0.40
        elif word_count <= 500:
            return 0.90
        elif word_count <= 1000:
            return 0.75
        else:
            return 0.55

    elif name == "structure":
        has_headers = any(line.startswith("#") for line in text.split("\n"))
        has_bullets = any(line.strip().startswith(("-", "*", "•")) for line in text.split("\n"))
        has_numbers = any(
            line.strip()[:2] in [f"{i}." for i in range(1, 10)] for line in text.split("\n")
        )
        has_paragraphs = text.count("\n\n") >= 1 or word_count > 50
        indicators = sum([has_headers, has_bullets, has_numbers, has_paragraphs])
        return min(1.0, indicators * 0.25)

    elif name == "specificity":
        numbers = len(re.findall(r"\b\d+[\.,]?\d*\b", text))
        code_blocks = text.count("```") // 2
        proper_nouns = sum(1 for w in text.split() if w and w[0].isupper() and len(w) > 3)
        return min(1.0, (numbers * 0.05) + (code_blocks * 0.15) + min(0.30, proper_nouns * 0.02))

    elif name == "actionability":
        action_words = [
            "instala",
            "ejecuta",
            "usa",
            "corre",
            "configura",
            "abre",
            "crea",
            "escribe",
            "modifica",
            "agrega",
            "pip",
            "conda",
            "install",
            "run",
            "execute",
            "configure",
            "create",
        ]
        matches = sum(1 for word in action_words if word in text.lower())
        return min(1.0, matches * 0.12)

    else:
        # Criterio genérico: combinación de longitud + estructura + números
        numbers = len(re.findall(r"\b\d+[\.,]?\d*\b", text))
        has_structure = any(
            line.startswith("#") or line.strip().startswith(("-", "*")) for line in text.split("\n")
        )
        base = min(0.70, word_count / 500.0)
        bonus = min(0.30, numbers * 0.03 + (0.10 if has_structure else 0.0))
        return min(1.0, base + bonus)


def score_heuristic(
    output: str,
    task_description: str = "",
    criteria: dict[str, float] | None = None,
    passing_threshold: float = 0.65,
) -> ScoreResult:
    """
    Evalúa un texto con reglas heurísticas sin usar un LLM adicional.

    Args:
        output: Texto a evaluar.
        task_description: Descripción de la tarea (para contextualizar el feedback).
        criteria: Dict {nombre_criterio: peso} con pesos que sumen 1.0.
                  Si es None, usa DEFAULT_CRITERIA.
        passing_threshold: Score mínimo (0–1) para considerar aprobado.

    Returns:
        ScoreResult con score (0–1), max_score (1.0), breakdown y feedback.
    """
    if criteria is None:
        criteria = DEFAULT_CRITERIA

    breakdown: dict[str, float] = {}
    feedback_parts: list[str] = []

    for name in criteria:
        dim_score = _score_dimension_heuristic(name, output)
        breakdown[name] = round(dim_score, 3)
        feedback_parts.append(f"{name}: {dim_score:.0%}")

    total_score = round(sum(breakdown[name] * weight for name, weight in criteria.items()), 4)

    feedback = (
        "Evaluación heurística"
        + (f" de '{task_description[:60]}'" if task_description else "")
        + f": {', '.join(feedback_parts)}. Score total: {total_score:.0%}."
    )

    return ScoreResult(
        score=total_score,
        max_score=1.0,
        breakdown=breakdown,
        feedback=feedback,
        passed=total_score >= passing_threshold,
    )


def score_with_llm(
    output: str,
    task_description: str = "",
    criteria: dict[str, float] | None = None,
    llm=None,
    passing_threshold: float = 0.65,
) -> ScoreResult:
    """
    Evalúa un texto usando un LLM como juez.

    Args:
        output: Texto a evaluar.
        task_description: Descripción de la tarea que generó el output.
        criteria: Dict {nombre_criterio: peso}. Si es None, usa DEFAULT_CRITERIA.
        llm: Instancia de LLM (crewai.LLM o LangChain ChatModel).
             Si es None, usa evaluación heurística como fallback.
        passing_threshold: Umbral para aprobar (0–1).

    Returns:
        ScoreResult con score, max_score, breakdown y feedback.
    """
    if criteria is None:
        criteria = DEFAULT_CRITERIA

    if llm is None:
        return score_heuristic(output, task_description, criteria, passing_threshold)

    criteria_lines = "\n".join(
        f'- "{name}" (peso {weight}): evalúa {name.replace("_", " ")} del texto'
        for name, weight in criteria.items()
    )
    names_json = ", ".join(f'"{name}": <score 0.0-1.0>' for name in criteria)

    prompt = (
        f"Actúa como evaluador experto. Evalúa el siguiente output según los criterios dados.\n"
        f"Da una puntuación de 0.0 a 1.0 para cada criterio (1.0 = perfecto, 0.0 = ausente).\n\n"
        f"TAREA EVALUADA: {task_description}\n\n"
        f"CRITERIOS:\n{criteria_lines}\n\n"
        f"OUTPUT A EVALUAR:\n{output[:3000]}\n\n"
        f"Responde ÚNICAMENTE con un JSON válido, sin texto adicional:\n"
        f"{{{names_json}}}"
    )

    import json

    try:
        # crewai.LLM → .call(messages: list[dict]) → str
        # LangChain ChatModel → .invoke(str) → objeto con .content
        if hasattr(llm, "call"):
            content = llm.call([{"role": "user", "content": prompt}])
            if not isinstance(content, str):
                content = str(content)
        elif hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
        else:
            raise AttributeError("LLM incompatible: necesita .call() o .invoke()")

        json_match = re.search(r"\{[^{}]+\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON encontrado en la respuesta del LLM.")

        raw_scores = json.loads(json_match.group())
        breakdown: dict[str, float] = {
            name: round(min(1.0, max(0.0, float(raw_scores.get(name, 0.5)))), 3)
            for name in criteria
        }

    # Cualquier fallo del LLM (parseo, red, formato de respuesta) debe caer
    # al fallback heurístico, nunca tumbar la evaluación.
    except Exception as exc:  # noqa: BLE001
        fallback = score_heuristic(output, task_description, criteria, passing_threshold)
        fallback.feedback = (
            f"[Fallback heurístico — error LLM: {type(exc).__name__}] " + fallback.feedback
        )
        return fallback

    total_score = round(sum(score * criteria[name] for name, score in breakdown.items()), 4)
    feedback_parts = [f"{name}: {val:.0%}" for name, val in breakdown.items()]
    feedback = (
        "Evaluación LLM"
        + (f" de '{task_description[:60]}'" if task_description else "")
        + f": {', '.join(feedback_parts)}. Score total: {total_score:.0%}."
    )

    return ScoreResult(
        score=total_score,
        max_score=1.0,
        breakdown=breakdown,
        feedback=feedback,
        passed=total_score >= passing_threshold,
    )
