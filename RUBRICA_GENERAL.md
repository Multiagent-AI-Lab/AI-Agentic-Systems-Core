# RÚBRICA GENERAL DE EVALUACIÓN DE CÓDIGO Y ENTREGABLES
## Sistemas de IA Agénticos — UCEMICH

Rúbrica de 5 criterios usada por `EvaluatorAgent` (`src/multiagent_core/evaluator_agent.py`) para calificar código de estudiantes. Los nombres de criterio y de nivel de esta tabla son exactos a los que produce el código — no descriptivos aproximados.

| Criterio | Descripción | Sobresaliente (95) | Competente (82) | En desarrollo (67) | Insuficiente (45) |
|---|---|---|---|---|---|
| **Corrección lógica** | El código es sintácticamente válido y resuelve el problema planteado | — (evaluación automática solo distingue Insuficiente/Competente; casos borde requieren revisión manual) | Compila sin errores de sintaxis | — | No compila (`SyntaxError`) |
| **Proceso (patrón pedagógico)** | Documentación explícita de las 6 fases de El Ciclo del Agente (Selección de Arquitectura, Diseño, Implementación, Evaluación, Despliegue, Iteración) | — | — | No verificable automáticamente a partir del código fuente — el estudiante debe documentar el ciclo completo en su entrega para que un revisor humano confirme este criterio | — |
| **Calidad de código** | Estilo PEP8 y ausencia de riesgos de seguridad OWASP, vía `CodeAuditorAgent` | 0 hallazgos de estilo o seguridad | Exactamente 1 hallazgo de estilo, ninguno de seguridad | 2+ hallazgos de estilo, ninguno de seguridad | Cualquier hallazgo de seguridad (independientemente de hallazgos de estilo) |
| **Pruebas (pytest)** | Existencia y resultado de un archivo de pruebas unitarias | — | El archivo de pruebas existe y todas las pruebas pasan | El archivo de pruebas existe pero al menos una prueba falla | No se proporcionó un archivo de pruebas pytest |
| **Reproducibilidad** | Determinismo del código o uso correcto de semilla aleatoria | El código es determinista — no usa `random`/`np.random`, no requiere semilla | Usa aleatoriedad (`random`/`np.random`) con una semilla fija (`seed(...)`) | — | Usa aleatoriedad sin fijar semilla — el resultado cambia en cada ejecución |

**Calificación final**: promedio simple de los 5 criterios, cada uno convertido a su puntaje de nivel (Sobresaliente=95, Competente=82, En desarrollo=67, Insuficiente=45).

## Notas de uso

- El criterio **Proceso** nunca se resuelve automáticamente en "Sobresaliente" ni "Insuficiente" — siempre retorna "En desarrollo" con la nota de que requiere verificación manual de un revisor humano. Esto es una limitación conocida y deliberada del agente, no un error: verificar que las 6 fases de El Ciclo del Agente estén genuinamente documentadas (no solo mencionadas) requiere criterio pedagógico que el análisis estático de código no puede aportar.
- El criterio **Calidad de código** reutiliza exactamente `CodeAuditorAgent.audit_style()` y `.audit_security()` — cualquier cambio a los patrones de esos métodos cambia automáticamente los umbrales de esta rúbrica.
- Un solo hallazgo de seguridad (p. ej. uso de `eval()`) fuerza "Insuficiente" en Calidad de código sin importar cuántos otros criterios de estilo se cumplan — la seguridad no es promediable con el estilo.
- Esta rúbrica califica **código de estudiantes**, no el contenido pedagógico de las unidades (`lecciones/*.md`) — para eso existe `ContentAuditorAgent`, que audita el material del curso en sí, no las entregas.
