"""src/multiagent_core/skills/agent_warmup/context_loader.py
Inyecta contexto de dominio en agentes LLM.
"""

from __future__ import annotations

SKILL_METADATA = {
    "name": "context_loader",
    "domain": "agent_warmup",
    "description": "Inyecta contexto de dominio en agentes LLM. Soporta 5 dominios predefinidos y contexto custom.",
    "version": "1.0.0",
    "input": "domain: str, custom_config: dict | None",
    "output": "dict con system_context, messages, domain",
    "dependencies": [],
}

_DOMAIN_CONTEXTS: dict[str, str] = {
    "research": (
        "Eres un asistente de investigación científica especializado en análisis de literatura, "
        "síntesis de conocimiento y generación de hipótesis. Aportas rigor metodológico y "
        "citas evidencia cuando es posible. Tu audiencia es investigadores de postgrado."
    ),
    "engineering": (
        "Eres un ingeniero de software senior especializado en arquitectura de sistemas, "
        "buenas prácticas de código y solución de problemas técnicos complejos. "
        "Prioriza soluciones mantenibles y bien documentadas."
    ),
    "data_analysis": (
        "Eres un analista de datos experto en estadística, visualización y extracción de "
        "insights accionables desde datasets complejos. Usas Python (pandas, numpy, matplotlib) "
        "y explicas resultados con claridad para audiencias no técnicas."
    ),
    "teaching": (
        "Eres un educador experto en diseño instruccional y aprendizaje activo. "
        "Adaptas la complejidad al nivel del estudiante, usas analogías efectivas y "
        "promueves el pensamiento crítico con preguntas guía."
    ),
    "nanotechnology": (
        "Eres un investigador experto en nanotecnología con conocimientos en DFT, "
        "dinámica molecular, síntesis de nanopartículas, caracterización (TEM, XRD, espectroscopía) "
        "y aplicaciones en nanomedicina y materiales. Conectas simulación computacional con "
        "resultados experimentales."
    ),
}


def warm_up(domain: str = "research", custom_config: dict | None = None) -> dict:
    """
    Genera el contexto de dominio para inicializar un agente LLM.

    Args:
        domain: Nombre del dominio. Uno de: research, engineering, data_analysis,
                teaching, nanotechnology, custom.
        custom_config: Requerido solo si domain='custom'. Dict con clave 'system_prompt'.

    Returns:
        Dict con:
            - system_context (str): Prompt de sistema con contexto de dominio.
            - messages (list[dict]): Lista de mensajes con formato {role, content}.
            - domain (str): Nombre del dominio aplicado.

    Raises:
        ValueError: Si el dominio no existe y no es 'custom', o si domain='custom'
                    sin proveer system_prompt en custom_config.
    """
    if domain == "custom":
        if custom_config is None or "system_prompt" not in custom_config:
            raise ValueError(
                "Para domain='custom' debes proveer custom_config con la clave 'system_prompt'."
            )
        system_context = custom_config["system_prompt"]
    elif domain in _DOMAIN_CONTEXTS:
        system_context = _DOMAIN_CONTEXTS[domain]
    else:
        available = list(_DOMAIN_CONTEXTS.keys()) + ["custom"]
        raise ValueError(f"Dominio '{domain}' no encontrado. Dominios disponibles: {available}")

    messages = [{"role": "system", "content": system_context}]

    return {
        "system_context": system_context,
        "messages": messages,
        "domain": domain,
    }


def list_available_domains() -> list[str]:
    """Retorna la lista de dominios predefinidos más 'custom'."""
    return list(_DOMAIN_CONTEXTS.keys()) + ["custom"]


def apply_to_agent(agent_instance: object, domain: str = "research") -> object:
    """
    Intenta inyectar el contexto de dominio en un agente existente.
    Compatible con objetos que tengan atributo 'system_message' o 'backstory'.

    Args:
        agent_instance: Instancia del agente a configurar.
        domain: Dominio de contexto a aplicar.

    Returns:
        El mismo agente_instance con el contexto aplicado.
    """
    ctx = warm_up(domain)
    system_context = ctx["system_context"]

    if hasattr(agent_instance, "system_message"):
        agent_instance.system_message = system_context
    elif hasattr(agent_instance, "backstory"):
        current = getattr(agent_instance, "backstory", "")
        agent_instance.backstory = f"{current}\n\n{system_context}".strip()

    return agent_instance
