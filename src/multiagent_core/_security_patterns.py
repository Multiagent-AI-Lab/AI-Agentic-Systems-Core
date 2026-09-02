"""
Patrones de detección de seguridad compartidos entre agentes.

Usado por CodeAuditorAgent, ContentAuditorAgent y SafetyGateAgent, que
antes cada uno declaraba su propia copia idéntica de este regex — este
módulo consolida esa única definición para que una futura corrección
(nuevos alias, formatos adicionales) se aplique en un solo lugar.
"""

import re

API_KEY_PATTERN = re.compile(
    r'(api_key|token|password|secret|key|passwd)\s*=\s*[\'"][a-zA-Z0-9_\-\.]{10,}[\'"]',
    re.IGNORECASE,
)
