# UNIDAD 0: Fundamentos Matemáticos

**Curso:** AI-Agentic-Systems-Core — UCEMICH

**Fuente:** Adaptado de "Fundamentos Matemáticos de la Inteligencia Artificial" (EMALCA Ecuador 2026, Yachay Tech). El contenido matemático de esta unidad se redacta guiado por esa fuente, no copiado literal: se conserva su rigor formal y su narrativa (cada componente de un sistema de IA resuelve un problema matemático concreto), pero se reescribe en prosa propia, se reordena para esta unidad y se agrega, por cada capítulo, un ejemplo de código ejecutado y verificado que la fuente original no tenía en ese formato.

## Prefacio

Esta unidad parte de una premisa deliberadamente distinta a la de un curso introductorio de programación: **si ya sabes álgebra lineal, cálculo multivariable y probabilidad, ya tienes la mayor parte de las herramientas que mueven a los sistemas de IA más avanzados que existen hoy**. Lo que falta no es matemática más sofisticada, sino ver cómo esas herramientas se ensamblan en una arquitectura computacional que percibe, razona y actúa — un **agente**.

Este curso no es un curso de matemáticas puras ni un curso de "aprende a llamar una API de LLM". Es un curso de **ingeniería de sistemas multiagente**, y esta unidad es su cimiento formal. Cada unidad posterior invocará explícitamente los conceptos aquí introducidos: el Diccionario de Variables de esta unidad reaparecerá en fórmulas de unidades futuras, y la sección de Teoría de la Información/Causalidad (Capítulo 6) se retoma explícitamente en la fase de Evaluación del Ciclo del Agente en la Unidad 3.

El objetivo no es demostrar teoremas desde los axiomas — es mostrar la correspondencia exacta entre la fórmula en papel y la línea de código que la implementa, y dejar claro qué pregunta matemática resuelve cada pieza de un agente de IA.

---

## Capítulo 1 — Álgebra Lineal y Álgebra Multilineal

### Motivación

Un modelo de lenguaje contemporáneo contiene entre $10^9$ y $10^{12}$ parámetros numéricos. Cada operación que realiza un agente construido sobre ese modelo —codificar una observación, comparar dos conceptos en su memoria vectorial, aplicar una capa de atención— es, en su núcleo, una secuencia de operaciones sobre matrices y vectores. El álgebra lineal no es una herramienta de apoyo para estos sistemas: es su columna vertebral.

### Definiciones

**Espacio vectorial.** Un espacio vectorial sobre $\mathbb{F}$ es un conjunto $V$ con suma y producto escalar que satisface los ocho axiomas estándar (asociatividad, existencia de neutro, de inverso aditivo, distributividad, etc.). Los ejemplos fundamentales para IA son $\mathbb{R}^n$ (vectores de features o embeddings) y $\mathbb{R}^{m\times n}$ (matrices de pesos).

**Transformación lineal.** $T: V \to W$ es lineal si $T(u+v)=T(u)+T(v)$ y $T(\alpha v)=\alpha T(v)$ para todos $u,v\in V$, $\alpha\in\mathbb{F}$. Para $V=\mathbb{R}^n$, $W=\mathbb{R}^m$, toda transformación lineal tiene la forma $T(x)=Ax$ para una única matriz $A\in\mathbb{R}^{m\times n}$: cada capa de una red neuronal es, antes de la no linealidad, exactamente esto.

**Valores y vectores propios.** $\lambda\in\mathbb{C}$ es un valor propio de $A\in\mathbb{R}^{n\times n}$ con vector propio $v\neq\mathbf{0}$ si $Av=\lambda v$.

### Descomposición en Valores Singulares (SVD)

**Fórmula (1.1):**
$$A = U\Sigma V^\top, \quad A\in\mathbb{R}^{m\times n},\ U\in\mathbb{R}^{m\times m},\ \Sigma\in\mathbb{R}^{m\times n},\ V\in\mathbb{R}^{n\times n}$$

$U$ y $V$ son matrices ortogonales; $\Sigma=\text{diag}(\sigma_1,\ldots,\sigma_{\min(m,n)})$ con $\sigma_1\geq\sigma_2\geq\cdots\geq 0$. Interpretación geométrica: toda transformación lineal es una rotación, seguida de un escalado a lo largo de ejes ortogonales, seguida de otra rotación.

**Teorema de Eckart–Young–Mirsky (mejor aproximación de bajo rango), fórmula (1.2):**
$$A_k = \sum_{i=1}^{k}\sigma_i u_i v_i^\top = U_k\Sigma_k V_k^\top, \qquad \|A-A_k\|_F^2 = \sum_{i=k+1}^{\min(m,n)}\sigma_i^2$$

Es decir: si truncamos la SVD a los $k$ valores singulares más grandes, obtenemos la mejor aproximación posible de rango $k$ en norma de Frobenius, y el error cometido es exactamente la raíz de la suma de los valores singulares descartados. Este resultado es la base matemática de PCA, de la compresión de matrices de atención, y de técnicas de fine-tuning eficiente como **LoRA** (fórmula 1.3):

$$W' = W_0 + \Delta W = W_0 + BA, \quad B\in\mathbb{R}^{m\times r},\ A\in\mathbb{R}^{r\times n},\ r\ll\min(m,n)$$

Solo $B$ y $A$ se entrenan; con $r=8$ en una matriz $4096\times4096$, el ahorro de parámetros entrenables es un factor de $\approx 256$.

### Código verificado: SVD y aproximación de bajo rango

El siguiente bloque construye una matriz aleatoria, calcula su SVD completa, verifica que reconstruye exactamente la matriz original, y luego verifica el Teorema de Eckart–Young–Mirsky comparando el error de una aproximación de rango 2 contra su predicción teórica.

```python
import numpy as np


def aproximar_rango_bajo(matriz: np.ndarray, rango: int) -> np.ndarray:
    """Reconstruye una aproximación de rango `rango` de `matriz` vía SVD truncada."""
    U, S, Vt = np.linalg.svd(matriz, full_matrices=False)
    return U[:, :rango] @ np.diag(S[:rango]) @ Vt[:rango, :]


np.random.seed(42)  # fija también el generador legado, por reproducibilidad explícita
rng = np.random.default_rng(42)
A = rng.normal(size=(6, 4))

U, S, Vt = np.linalg.svd(A)
reconstruccion_completa = U[:, :4] @ np.diag(S) @ Vt
assert np.allclose(A, reconstruccion_completa), "La SVD completa debe reconstruir A"
print(f"Valores singulares: {np.round(S, 4)}")

A_2 = aproximar_rango_bajo(A, 2)
error_frobenius = np.linalg.norm(A - A_2, ord="fro")
error_teorico = np.sqrt(np.sum(S[2:] ** 2))
print(f"Error Frobenius ||A - A_2||: {error_frobenius:.6f}")
print(f"Error teorico Eckart-Young (sqrt(sum sigma_i^2, i>k)): {error_teorico:.6f}")
assert np.isclose(error_frobenius, error_teorico), "Eckart-Young-Mirsky debe cumplirse"
assert np.linalg.matrix_rank(A_2) == 2
```

Ejecutado, este bloque imprime `Valores singulares: [3.0261 2.1438 1.5253 0.5876]` y confirma con `assert` que el error de la aproximación de rango 2 (`1.634541`) coincide, hasta precisión numérica, con la predicción del teorema.

### Conexión con sistemas de IA actuales

| Concepto matemático | Sistema IA | Rol concreto |
|---|---|---|
| SVD y bajo rango | PCA, compresión de modelos | Reducir dimensionalidad; comprimir matrices de atención |
| LoRA | Fine-tuning de LLMs | Adaptar modelos masivos con una sola GPU de consumo |
| Red neuronal como composición de $y=Wx+b$ | Toda red neuronal profunda | La arquitectura entera se reduce a esta fórmula, capa a capa |
| Embeddings en $\mathbb{R}^d$ | Bases de datos vectoriales (memoria de agentes) | Similitud semántica como proximidad geométrica |

---

## Capítulo 2 — Cálculo Multivariado y Backpropagation

### Motivación

Entrenar una red neuronal significa encontrar parámetros $\theta$ que minimicen una función de pérdida $L(\theta)$ que puede depender de miles de millones de coordenadas. La pregunta central es: ¿cómo calculamos, de forma eficiente, cómo cambia $L$ respecto a cada parámetro individual? La respuesta es *backpropagation*, que no es más que la regla de la cadena del cálculo multivariado aplicada con inteligencia computacional sobre un grafo de cómputo.

### Definiciones

Una red neuronal de $\ell$ capas define una función compuesta $F(\mathbf{x};\theta)=f_\ell\circ f_{\ell-1}\circ\cdots\circ f_1(\mathbf{x})$, donde cada capa es $f_k(\mathbf{z})=\sigma(\mathbf{W}_k\mathbf{z}+\mathbf{b}_k)$ con $\sigma$ una no linealidad.

### Fórmulas clave

**(2.1) Gradiente de la pérdida respecto a los pesos de la capa $\ell$ (regla de la cadena, propagada hacia atrás):**
$$\frac{\partial L}{\partial \mathbf{W}_\ell} = \boldsymbol{\delta}_\ell\cdot\mathbf{x}_{\ell-1}^\top, \qquad \boldsymbol{\delta}_{\ell-1} = \left(\mathbf{W}_\ell^\top\boldsymbol{\delta}_\ell\right)\odot\sigma'(\mathbf{z}_{\ell-1})$$

**(2.2) Descenso de gradiente estocástico (SGD):** $\theta_{t+1}=\theta_t-\eta_t\nabla_\theta L(\theta_t;\mathcal{B}_t)$

**(2.3) Condición de punto crítico:** $\nabla L(\theta^*)=0$. En dimensión alta ($d\sim10^{11}$), la probabilidad de que un punto crítico sea un mínimo local verdadero es $\approx(1/2)^d\approx 0$: los puntos críticos de alta pérdida son casi siempre **puntos de silla** (Dauphin et al., 2014), no mínimos locales — una de las razones por las que SGD con ruido de mini-batch funciona bien en la práctica.

### Backpropagation como diferenciación automática

*Backpropagation* es la regla de la cadena aplicada al grafo computacional, recorrido de atrás hacia adelante. La diferenciación automática en modo reverso implementa esto con complejidad $O(|\theta|)$ en tiempo y memoria — eficiente precisamente porque la pérdida es escalar ($m=1$) mientras $|\theta|$ puede ser enorme.

### Código verificado: backpropagation manual contra diferenciación numérica

Para no tratar backpropagation como una caja negra, el siguiente bloque implementa a mano el forward pass y el gradiente (vía regla de la cadena) de una red diminuta de dos capas $z_1=\tanh(W_1x)$, $y=w_2\cdot z_1$, $L=y^2$, y verifica el resultado contra diferenciación numérica por diferencias finitas centradas — el método más directo e independiente de comprobar que una derivada analítica es correcta.

```python
import numpy as np


def forward_y_gradiente(
    x: np.ndarray, w1: np.ndarray, w2: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Red de 2 capas z1 = tanh(W1 x), y = w2 . z1, perdida L = y^2.

    Retorna (perdida, gradiente respecto a W1, gradiente respecto a w2)
    calculados a mano con la regla de la cadena (backpropagation).
    """
    z1_pre = w1 @ x
    z1 = np.tanh(z1_pre)
    y = w2 @ z1
    perdida = y**2

    d_perdida_d_y = 2.0 * y
    delta1 = d_perdida_d_y * w2 * (1.0 - z1**2)
    grad_w1 = np.outer(delta1, x)
    grad_w2 = d_perdida_d_y * z1
    return perdida, grad_w1, grad_w2


def perdida_escalar(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> float:
    """Evalua solo la perdida escalar L(W1, w2) = (w2 . tanh(W1 x))^2."""
    z1 = np.tanh(w1 @ x)
    y = w2 @ z1
    return float(y**2)


np.random.seed(7)  # fija también el generador legado, por reproducibilidad explícita
rng = np.random.default_rng(7)
x = rng.normal(size=3)
w1 = rng.normal(size=(2, 3))
w2 = rng.normal(size=2)

perdida, grad_w1_analitico, grad_w2_analitico = forward_y_gradiente(x, w1, w2)

eps = 1e-6
grad_w1_numerico = np.zeros_like(w1)
for i in range(w1.shape[0]):
    for j in range(w1.shape[1]):
        w1_mas, w1_menos = w1.copy(), w1.copy()
        w1_mas[i, j] += eps
        w1_menos[i, j] -= eps
        grad_w1_numerico[i, j] = (
            perdida_escalar(x, w1_mas, w2) - perdida_escalar(x, w1_menos, w2)
        ) / (2 * eps)

assert np.allclose(grad_w1_analitico, grad_w1_numerico, atol=1e-4)
print(f"Perdida: {perdida:.6f}")
print(f"grad_w1 (backprop): {np.round(grad_w1_analitico, 5)}")
```

Ejecutado, confirma `Perdida: 0.024512` y que el gradiente analítico calculado con la regla de la cadena coincide, hasta `1e-4`, con el gradiente numérico — la misma verificación que usa cualquier framework de autograd (`torch.autograd.gradcheck`) para certificar una nueva operación diferenciable.

### Conexión con sistemas de IA actuales

| Concepto | Dónde aparece | Ejemplo |
|---|---|---|
| Regla de la cadena | Autograd en PyTorch/JAX | `loss.backward()` recorre el grafo en reversa |
| SGD con mini-batches | Entrenamiento de LLMs | Variantes de AdamW en modelos de producción |
| Dominancia de puntos de silla | Explica por qué SGD escapa mínimos falsos | El ruido del mini-batch perturba la silla |

---

## Capítulo 3 — Sistemas Dinámicos, Caos y Neural ODEs

### Motivación

Una red neuronal puede verse como una función estática, pero esa perspectiva oculta la riqueza dinámica del entrenamiento y de arquitecturas recurrentes. ¿Qué ocurre cuando una red procesa secuencias largas, o cuando se apilan cientos de capas? La teoría de sistemas dinámicos da el lenguaje preciso para responder.

### Definiciones

Un **sistema dinámico continuo** es $\dot{\mathbf{z}}(t)=\mathbf{f}(\mathbf{z}(t),t)$, $\mathbf{z}(0)=\mathbf{z}_0$. El punto fijo $\mathbf{z}^*$ es **estable** (Lyapunov) si trayectorias que comienzan cerca de $\mathbf{z}^*$ permanecen cerca para todo $t\geq 0$.

### Exponentes de Lyapunov y caos

**Fórmula (3.1):**
$$\lambda = \lim_{n\to\infty}\frac{1}{n}\sum_{k=0}^{n-1}\log|f'(z_k)|$$

- $\lambda>0$: caos — sensibilidad exponencial a las condiciones iniciales.
- $\lambda<0$: convergencia estable a un punto fijo o ciclo.
- $\lambda=0$: frontera del caos (*edge of chaos*).

En redes recurrentes con pesos $W_{ij}\sim\mathcal{N}(0,g^2/N)$ (Sompolinsky et al., 1988), $g<1$ produce un único punto fijo estable, $g>1$ produce dinámica caótica —origen práctico del problema de gradientes explosivos/evanescentes en RNNs.

### ResNets como ecuaciones diferenciales discretizadas

Una capa residual $\mathbf{z}_{k+1}=\mathbf{z}_k+h\cdot\mathbf{f}(\mathbf{z}_k,\theta_k)$ es exactamente el método de Euler explícito para integrar $\dot{\mathbf{z}}=\mathbf{f}(\mathbf{z},t)$. Las **Neural ODEs** (Chen et al., 2018) llevan esto al límite continuo, calculando gradientes con memoria $O(1)$ vía el **método del estado adjunto**:
$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^\top\cdot\frac{\partial\mathbf{f}}{\partial\mathbf{z}}(\mathbf{z}(t),t,\theta)$$

### Código verificado: exponente de Lyapunov del mapa logístico

En lugar de reproducir el sistema de Lorenz (tridimensional, requiere un integrador numérico completo para un solo número que verificar), este bloque calcula el exponente de Lyapunov del **mapa logístico** $x_{n+1}=rx_n(1-x_n)$ — el ejemplo unidimensional canónico de transición al caos — y confirma que el signo de $\lambda$ distingue el régimen estable del caótico, exactamente como predice la fórmula (3.1).

```python
import numpy as np


def exponente_lyapunov_logistico(r: float, n_iter: int, x0: float = 0.4) -> float:
    """Estima el exponente de Lyapunov del mapa logistico x_{n+1}=r*x_n*(1-x_n).

    lambda = lim (1/n) * suma log|f'(x_k)|, con f'(x) = r*(1-2x).
    """
    x = x0
    suma_log = 0.0
    for _ in range(n_iter):
        derivada = r * (1.0 - 2.0 * x)
        suma_log += np.log(abs(derivada))
        x = r * x * (1.0 - x)
    return suma_log / n_iter


lambda_estable = exponente_lyapunov_logistico(r=2.5, n_iter=5000)
lambda_caotico = exponente_lyapunov_logistico(r=3.9, n_iter=5000)

print(f"lambda(r=2.5) = {lambda_estable:.4f}  (regimen de punto fijo)")
print(f"lambda(r=3.9) = {lambda_caotico:.4f}  (regimen caotico)")
assert lambda_estable < 0, "r=2.5 debe converger a un punto fijo estable"
assert lambda_caotico > 0, "r=3.9 debe ser caotico"
```

Ejecutado, produce `lambda(r=2.5) = -0.6931` (estable) y `lambda(r=3.9) = 0.5027` (caótico), confirmando con `assert` la predicción teórica de signo.

### Conexión con sistemas de IA actuales

| Concepto | Dónde aparece | Ejemplo |
|---|---|---|
| Exponentes de Lyapunov | Diagnóstico de RNNs | $\lambda>0$ señala gradiente explosivo |
| ResNet $\approx$ Euler explícito | Redes muy profundas | Entrenar cientos de capas sin degradación |
| Neural ODEs | Series temporales continuas | Modelos con observaciones irregulares en el tiempo |

---

## Capítulo 4 — Probabilidad, Procesos Estocásticos e Inferencia

### Motivación

La IA moderna no es determinista. Un modelo de lenguaje produce una distribución sobre respuestas posibles; un agente que decide qué herramienta invocar lo hace, en muchos diseños, muestreando de una política estocástica. La tesis central de este capítulo: **toda la inferencia estadística en IA es un problema de aproximar distribuciones intratables**.

### Definiciones

Un espacio de probabilidad es la terna $(\Omega,\mathcal{F},P)$: $\Omega$ el espacio muestral, $\mathcal{F}$ una σ-álgebra de eventos, $P:\mathcal{F}\to[0,1]$ una medida con $P(\Omega)=1$ y aditividad numerable.

### Fórmulas clave

**(4.1) Teorema de Bayes:** $p(\theta\mid D)=\dfrac{p(D\mid\theta)p(\theta)}{p(D)}$

**(4.2) Estimador de máxima verosimilitud (MLE):** $\theta^*=\operatorname{argmax}_\theta\frac{1}{n}\sum_{i=1}^n\log p(x_i;\theta)$ — minimizar la entropía cruzada en un clasificador **es** maximizar la log-verosimilitud.

**(4.3) Estimador MAP:** $\theta_{\text{MAP}}=\operatorname{argmax}_\theta\left[\log p(D\mid\theta)+\log p(\theta)\right]$. Un resultado importante y no siempre explicitado: un prior gaussiano $p(\theta)=\mathcal{N}(0,\lambda^{-1}I)$ hace que el MAP coincida exactamente con la regularización $L_2$ (*weight decay*) que se usa rutinariamente al entrenar redes neuronales — no son dos técnicas distintas, son la misma estimación bayesiana vista desde dos idiomas.

**(4.4) ELBO del autoencoder variacional (VAE):**
$$\mathcal{L}(\phi,\theta) = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right] - D_{\text{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\right)$$

### Código verificado: MAP gaussiano es regresión ridge

El siguiente bloque calcula el estimador MAP en forma cerrada para un modelo lineal $y=X\theta+\varepsilon$ con prior gaussiano sobre $\theta$, y verifica —minimizando el negativo log-posterior de forma completamente independiente con un optimizador numérico (`scipy.optimize.minimize`)— que ambos caminos llegan al mismo punto, confirmando la equivalencia MAP-gaussiano $\equiv$ ridge de la fórmula (4.3).

```python
import numpy as np
from scipy.optimize import minimize


def estimador_map_gaussiano(
    X: np.ndarray, y: np.ndarray, lambda_reg: float
) -> np.ndarray:
    """Estimador MAP de theta para y = X @ theta + ruido gaussiano, con
    prior p(theta) = N(0, lambda_reg^-1 * I). Solucion cerrada (ridge):
    theta_MAP = (X^T X + lambda_reg * I)^-1 X^T y
    """
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + lambda_reg * np.eye(n_features), X.T @ y)


np.random.seed(3)  # fija también el generador legado, por reproducibilidad explícita
rng = np.random.default_rng(3)
n_muestras, n_features = 30, 4
X = rng.normal(size=(n_muestras, n_features))
theta_verdadero = np.array([1.5, -2.0, 0.5, 3.0])
y = X @ theta_verdadero + 0.1 * rng.normal(size=n_muestras)

lambda_reg = 2.0
theta_map = estimador_map_gaussiano(X, y, lambda_reg)


def negativo_log_posterior(theta: np.ndarray) -> float:
    """Negativo log-posterior (hasta constante aditiva) para theta."""
    residuo = y - X @ theta
    return float(residuo @ residuo + lambda_reg * (theta @ theta))


resultado = minimize(negativo_log_posterior, x0=np.zeros(n_features), method="BFGS")
print(f"theta_MAP (forma cerrada):      {np.round(theta_map, 4)}")
print(f"theta_MAP (optimizado BFGS):    {np.round(resultado.x, 4)}")
assert np.allclose(theta_map, resultado.x, atol=1e-4)
```

Ejecutado, ambos caminos convergen a `[1.4476 -1.9498 0.5364 2.7552]`, confirmando la equivalencia.

### Conexión con sistemas de IA actuales

| Concepto | Sistema IA | Cómo aparece |
|---|---|---|
| MLE | Cualquier red neuronal | Minimizar cross-entropy = maximizar log-verosimilitud |
| MAP $\equiv$ ridge | Regularización en producción | Weight decay $L_2$ es un prior gaussiano implícito |
| ELBO | VAE, VQ-VAE | Balance entre reconstrucción y regularización |
| Proceso de difusión | Modelos generativos de imagen/video | Destrucción y reconstrucción como proceso estocástico |

---

## Capítulo 5 — Teoría de la Información, Causalidad y XAI

### Motivación

En 1948, Claude Shannon preguntó cuánta información contiene un mensaje. La respuesta —la entropía— se convirtió en el idioma en que se escriben las funciones de pérdida de toda red neuronal y en el fundamento cuantitativo del debate sobre interpretabilidad (*XAI*, explainable AI) y sobre causalidad: ¿qué le pasaría a la salida de un sistema si interviniéramos sobre una de sus entradas?

### Fórmulas clave

**(5.1) Entropía de Shannon:** $H(X)=-\sum_x p(x)\log p(x)\geq 0$

**(5.2) Entropía cruzada:** $H(p,q)=-\sum_x p(x)\log q(x)$ — la función de pérdida estándar en clasificación.

**(5.3) Divergencia de Kullback-Leibler:** $D_{\text{KL}}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q)-H(p)\geq 0$. No es simétrica: $D_{\text{KL}}(p\|q)\neq D_{\text{KL}}(q\|p)$.

**(5.4) Información mutua:** $I(X;Y)=H(X)-H(X\mid Y)=D_{\text{KL}}\big(p(X,Y)\,\|\,p(X)p(Y)\big)$

### XAI: Valores de Shapley

**Fórmula (5.5):**
$$\phi_i(v) = \sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\left[v(S\cup\{i\})-v(S)\right]$$

Cada $\phi_i$ es la contribución marginal promedio de la variable $i$, promediada sobre todos los órdenes posibles en que las variables podrían "entrar" a la coalición. Cumple la **propiedad de eficiencia**: $\sum_i\phi_i = v(N)-v(\emptyset)$ — la predicción completa se reparte exactamente entre las features, sin sobrante ni faltante. Es la base matemática de SHAP, la técnica de interpretabilidad más usada en modelos de producción.

### Causalidad: del grafo a la intervención

Correlación no es causalidad: $I(X;Y)>0$ solo dice que $X$ y $Y$ comparten información, no en qué dirección va el efecto. Un **grafo acíclico dirigido causal** (DAG causal) declara explícitamente esa dirección: cada arista $A\to B$ afirma que intervenir sobre $A$ puede cambiar $B$, no al revés. Esta distinción —observar $p(Y\mid X=x)$ frente a intervenir $p(Y\mid \text{do}(X=x))$— es la pregunta que un sistema de XAI causal necesita responder, y es la que la Unidad 3 retomará explícitamente al construir la fase de Evaluación del Ciclo del Agente: un agente que solo reporta correlaciones entre sus acciones y sus resultados no está explicando su comportamiento, solo lo está describiendo.

### Código verificado: entropía cruzada, KL, Shapley y un DAG causal

```python
import itertools
import math

import networkx as nx
import numpy as np


def entropia_cruzada(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula H(p,q) = -sum p(x) log q(x)."""
    return float(-np.sum(p * np.log(q)))


def divergencia_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula D_KL(p||q) = sum p(x) log(p(x)/q(x))."""
    return float(np.sum(p * np.log(p / q)))


p = np.array([0.7, 0.2, 0.1])
q = np.array([0.6, 0.3, 0.1])
h_pq = entropia_cruzada(p, q)
h_p = entropia_cruzada(p, p)
kl_pq = divergencia_kl(p, q)
assert math.isclose(kl_pq, h_pq - h_p, abs_tol=1e-9)  # D_KL(p||q) = H(p,q) - H(p)
assert kl_pq >= 0.0
print(f"H(p,q)={h_pq:.5f}  H(p)={h_p:.5f}  D_KL(p||q)={kl_pq:.5f}")


def valor_shapley(
    jugadores: list[str], funcion_valor: dict[frozenset, float]
) -> dict[str, float]:
    """Calcula los valores de Shapley exactos para un juego cooperativo con
    funcion de valor v(S) dada explicitamente para cada coalicion S."""
    n = len(jugadores)
    phi = {jugador: 0.0 for jugador in jugadores}
    for jugador in jugadores:
        resto = [j for j in jugadores if j != jugador]
        for tam in range(len(resto) + 1):
            for subconjunto in itertools.combinations(resto, tam):
                s = frozenset(subconjunto)
                peso = (
                    math.factorial(len(s)) * math.factorial(n - len(s) - 1)
                ) / math.factorial(n)
                contribucion = funcion_valor[s | {jugador}] - funcion_valor[s]
                phi[jugador] += peso * contribucion
    return phi


jugadores = ["A", "B", "C"]
v = {
    frozenset(): 0.0, frozenset({"A"}): 1.0, frozenset({"B"}): 1.0,
    frozenset({"C"}): 0.0, frozenset({"A", "B"}): 4.0,
    frozenset({"A", "C"}): 2.0, frozenset({"B", "C"}): 2.0,
    frozenset({"A", "B", "C"}): 6.0,
}
phi = valor_shapley(jugadores, v)
suma_phi = sum(phi.values())
assert math.isclose(suma_phi, v[frozenset(jugadores)] - v[frozenset()], abs_tol=1e-9)
print(f"Valores de Shapley: { {k: round(val, 4) for k, val in phi.items()} }")

dag_causal = nx.DiGraph()
dag_causal.add_edges_from(
    [("Confusor", "Tratamiento"), ("Confusor", "Resultado"), ("Tratamiento", "Resultado")]
)
assert nx.is_directed_acyclic_graph(dag_causal)
padres_de_resultado = set(dag_causal.predecessors("Resultado"))
assert padres_de_resultado == {"Confusor", "Tratamiento"}
print(f"Padres causales de 'Resultado': {padres_de_resultado}")
```

Ejecutado, confirma la identidad $D_{\text{KL}}=H(p,q)-H(p)$, produce `Valores de Shapley: {'A': 2.5, 'B': 2.5, 'C': 1.0}` con suma exacta `6.0 = v(N)-v(∅)` (propiedad de eficiencia verificada por `assert`), y construye un DAG causal de 3 nodos confirmando con `assert` que es acíclico y que `Resultado` tiene como padres causales tanto al `Confusor` como al `Tratamiento` — el patrón estructural mínimo de un confusor en inferencia causal.

### Conexión con sistemas de IA actuales

| Concepto | Sistema IA | Cómo aparece |
|---|---|---|
| Entropía cruzada | Todo clasificador neuronal | Función de pérdida por defecto |
| Divergencia KL | VAE, RLHF | Término de regularización frente a una política de referencia |
| SHAP (Shapley) | Modelos de crédito, diagnóstico médico | Explicación aditiva de una predicción individual |
| DAG causal | Auditoría de agentes | Distinguir qué acción *causó* un resultado de qué solo *correlaciona* con él |

---

## Capítulo 6 — La Geometría del Mecanismo Transformer

### Motivación

Considera la oración: *"El banco estaba lleno de gente que esperaba cobrar, pero el pato nadaba junto al banco del río."* La palabra *banco* aparece dos veces con significados distintos. La pregunta matemática es cómo resolver esa ambigüedad de forma diferenciable y paralelizable — la respuesta es el mecanismo de **atención**.

### Fórmulas clave

**(6.1) Atención escalada (Scaled Dot-Product Attention):**
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $Q=XW^Q$: consultas — qué busca cada token.
- $K=XW^K$: claves — qué ofrece cada token.
- $V=XW^V$: valores — qué transmite cada token si es seleccionado.
- $QK^\top\in\mathbb{R}^{n\times n}$: similitud entre todos los pares de tokens.
- $/\sqrt{d_k}$: mantiene la varianza acotada para que el softmax no sature y anule el gradiente.

**(6.2) Multi-Head Attention:** $\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$, luego $\text{MHA}=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$. Cada cabeza puede especializarse en un tipo distinto de relación entre tokens.

**(6.3) Capa Transformer completa:** $x_\ell'=\text{LayerNorm}(x_\ell+\text{MHA}(x_\ell))$, $x_{\ell+1}=\text{LayerNorm}(x_\ell'+\text{FFN}(x_\ell'))$

### Código verificado: atención escalada como combinación convexa

```python
import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax numericamente estable aplicado por filas de z."""
    z_estable = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_estable)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def atencion_escalada(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.

    Retorna (salida, matriz_atencion) para inspeccionar los pesos.
    """
    d_k = Q.shape[-1]
    puntuaciones = (Q @ K.T) / np.sqrt(d_k)
    matriz_atencion = softmax(puntuaciones)
    salida = matriz_atencion @ V
    return salida, matriz_atencion


np.random.seed(11)  # fija también el generador legado, por reproducibilidad explícita
rng = np.random.default_rng(11)
n_tokens, d_k, d_v = 4, 8, 6
Q = rng.normal(size=(n_tokens, d_k))
K = rng.normal(size=(n_tokens, d_k))
V = rng.normal(size=(n_tokens, d_v))

salida, A = atencion_escalada(Q, K, V)
assert np.allclose(A.sum(axis=1), 1.0)  # cada fila de A es una distribucion de prob.
assert np.all(A >= 0.0)
for i in range(n_tokens):
    combinacion_manual = sum(A[i, j] * V[j] for j in range(n_tokens))
    assert np.allclose(combinacion_manual, salida[i], atol=1e-10)

print(f"Forma de la salida: {salida.shape}")
print(f"Suma de cada fila de A (debe ser 1.0): {np.round(A.sum(axis=1), 6)}")
```

Ejecutado, confirma que cada fila de la matriz de atención suma exactamente `1.0` (es una distribución de probabilidad) y que la salida de atención para cada token es, verificado término a término, la combinación convexa de los vectores `V` ponderada por esos pesos — exactamente lo que la fórmula (6.1) predice.

### Conexión con sistemas de IA actuales

| Componente | Modelo | Rol concreto |
|---|---|---|
| Scaled Dot-Product Attention | GPT-4, Llama 3, Claude | Mecanismo de atención central en cada capa |
| Multi-Head Attention | Modelos de producción actuales | Cabezas especializadas en relaciones distintas |
| Conexiones residuales + LayerNorm | Todos los Transformers | Permiten entrenar redes de 96+ capas |

---

## Capítulo 7 — Geometría Diferencial, Optimización Natural y Transporte Óptimo

### Motivación

El descenso por gradiente estándar ignora una pregunta: ¿el espacio de parámetros tiene curvatura? ¿Existe una noción de distancia más apropiada que la euclidiana para moverse en el espacio de modelos probabilísticos? La geometría de la información responde que sí: ese espacio tiene una geometría riemanniana natural, dada por la **métrica de Fisher**.

### Definiciones

**Variedad diferenciable** de dimensión $d$: espacio localmente homeomorfo a $\mathbb{R}^d$ con estructura diferenciable. **Hipótesis de la variedad:** datos reales de alta dimensión se concentran cerca de una subvariedad de dimensión mucho menor que el espacio ambiente (rostros humanos, por ejemplo, viven en una variedad de dimensión $\sim100$ dentro de un espacio de millones de píxeles).

### Fórmulas clave

**(7.1) Métrica de Fisher:**
$$F_{ij}(\theta) = \mathbb{E}_{x\sim p(x;\theta)}\!\left[\frac{\partial\log p(x;\theta)}{\partial\theta_i}\cdot\frac{\partial\log p(x;\theta)}{\partial\theta_j}\right]$$

Mide cuánto distingue una pequeña perturbación de $\theta$ desde el punto de vista de la distribución de datos que el modelo genera — no desde el punto de vista de la aritmética plana de $\mathbb{R}^p$.

**(7.2) Gradiente natural (Amari, 1998):** $\theta\leftarrow\theta-\eta\,F(\theta)^{-1}\nabla_\theta L(\theta)$ — invariante bajo reparametrizaciones, a diferencia del gradiente euclidiano estándar.

**(7.3) Distancia de Wasserstein-2:** $W_2^2(\mu,\nu)=\inf_{\gamma\in\Gamma(\mu,\nu)}\iint\|x-y\|^2\,d\gamma(x,y)$ — a diferencia de la KL, tiene sentido geométrico incluso entre distribuciones con soportes disjuntos, y es la base matemática de los modelos de difusión (esquema JKO) y de WGAN.

### Código verificado: métrica de Fisher y gradiente natural

Para una gaussiana $N(\mu,\sigma^2)$ con $\sigma$ fija, la fórmula cerrada de la información de Fisher respecto a $\mu$ es $F(\mu)=1/\sigma^2$. El siguiente bloque la estima también por Monte Carlo —muestreando la definición (7.1) directamente— y confirma que ambos caminos coinciden, y luego muestra numéricamente que el gradiente natural reescala el paso de optimización por $F(\mu)^{-1}$.

```python
import numpy as np


def informacion_fisher_media_gaussiana(sigma: float) -> float:
    """Informacion de Fisher F(mu) para el parametro mu de N(mu, sigma^2)
    con sigma conocida. Formula cerrada estandar: F(mu) = 1/sigma^2."""
    return 1.0 / sigma**2


def informacion_fisher_montecarlo(
    mu: float, sigma: float, n_muestras: int, rng: np.random.Generator
) -> float:
    """Estima F(mu) = E[(d/dmu log p(x;mu))^2] por muestreo Monte Carlo.

    Para N(mu, sigma^2): d/dmu log p(x;mu) = (x - mu) / sigma^2.
    """
    x = rng.normal(loc=mu, scale=sigma, size=n_muestras)
    score = (x - mu) / sigma**2
    return float(np.mean(score**2))


sigma, mu = 2.0, 1.5
np.random.seed(0)  # fija también el generador legado, por reproducibilidad explícita
rng = np.random.default_rng(0)
fisher_cerrada = informacion_fisher_media_gaussiana(sigma)
fisher_montecarlo = informacion_fisher_montecarlo(mu, sigma, 2_000_000, rng)
assert np.isclose(fisher_cerrada, fisher_montecarlo, rtol=0.02)

objetivo, mu_actual = 5.0, 0.0
gradiente_euclidiano = mu_actual - objetivo
gradiente_natural = (1.0 / fisher_cerrada) * gradiente_euclidiano
assert np.isclose(gradiente_natural, 4.0 * gradiente_euclidiano)  # F(mu)=0.25 -> factor 4

print(f"F(mu) formula cerrada: {fisher_cerrada:.5f}  |  Monte Carlo: {fisher_montecarlo:.5f}")
print(f"Gradiente euclidiano: {gradiente_euclidiano}  |  Gradiente natural: {gradiente_natural}")
```

Ejecutado, confirma `F(mu)=0.25` por ambos caminos (`0.25` vs `0.24991` con 2 millones de muestras) y que el gradiente natural amplifica el paso euclidiano por exactamente $\sigma^2 = 1/F(\mu) = 4$.

### Conexión con sistemas de IA actuales

| Concepto | Modelo | Rol |
|---|---|---|
| Gradiente natural | K-FAC y aproximaciones de curvatura | Optimización que respeta la geometría del espacio de modelos |
| Distancia de Wasserstein-2 | WGAN | Reemplaza la KL como función de pérdida entre distribuciones |
| Esquema JKO | Modelos de difusión | La difusión es un flujo de gradiente en el espacio de Wasserstein |

---

## Capítulo 8 — Topología Algebraica y Análisis de Datos

### Motivación

La geometría mide distancias y ángulos; la topología estudia qué es invariante bajo deformaciones continuas. Preguntas como "¿cuántos grupos separados tiene esta nube de datos?" o "¿tiene la variedad de datos agujeros de dimensión alta?" son preguntas topológicas, no geométricas.

### Fórmulas clave

**(8.1) Operador frontera:** $\partial_k[v_0,\ldots,v_k]=\sum_{i=0}^k(-1)^i[v_0,\ldots,\hat{v}_i,\ldots,v_k]$, con la propiedad fundamental $\partial_{k-1}\circ\partial_k=0$: "la frontera de una frontera es vacía".

**(8.2) Grupos de homología:** $H_k(K;\mathbb{F})=\ker(\partial_k)/\operatorname{im}(\partial_{k+1})$

**(8.3) Números de Betti:** $\beta_k=\dim_\mathbb{F}H_k(K;\mathbb{F})$

| $\beta_k$ | Significado | Ejemplo |
|---|---|---|
| $\beta_0$ | Componentes conexas | 3 grupos separados $\Rightarrow\beta_0=3$ |
| $\beta_1$ | Ciclos no contraíbles (agujeros 1-dimensionales) | Un círculo $\Rightarrow\beta_1=1$ |
| $\beta_2$ | Cavidades cerradas | Una esfera hueca $\Rightarrow\beta_2=1$ |

**(8.4) Estabilidad de TDA:** $d_B(\text{Dgm}(f),\text{Dgm}(g))\leq\|f-g\|_\infty$ — perturbaciones pequeñas en los datos producen perturbaciones pequeñas en los invariantes topológicos calculados, una garantía de robustez estadística poco común en análisis de datos de alta dimensión.

### Código verificado: números de Betti de un grafo vía el operador frontera

Un grafo es un complejo simplicial 1-dimensional: vértices (0-símplices) y aristas (1-símplices), sin caras 2-dimensionales. El siguiente bloque construye el operador frontera $\partial_1:C_1\to C_0$ como matriz, calcula $\beta_0$ y $\beta_1$ a partir de su rango (fórmulas 8.2–8.3 especializadas a este caso), y verifica el resultado contra `networkx` (componentes conexas y base de ciclos) — dos algoritmos completamente distintos que deben coincidir si la implementación del operador frontera es correcta.

```python
import networkx as nx
import numpy as np


def betti_0_y_1(
    vertices: list[int], aristas: list[tuple[int, int]]
) -> tuple[int, int]:
    """Calcula beta_0 (componentes conexas) y beta_1 (ciclos independientes)
    de un complejo simplicial 1-dimensional via el operador frontera
    boundary_1 : C_1 -> C_0.

    beta_0 = n_vertices - rango(boundary_1)
    beta_1 = n_aristas - rango(boundary_1)
    (no hay 2-simplices, asi que boundary_2 = 0 y su imagen es trivial)
    """
    n_v, n_e = len(vertices), len(aristas)
    indice = {v: i for i, v in enumerate(vertices)}

    boundary_1 = np.zeros((n_v, n_e))
    for j, (a, b) in enumerate(aristas):
        boundary_1[indice[a], j] = -1.0
        boundary_1[indice[b], j] = 1.0

    rango = np.linalg.matrix_rank(boundary_1)
    return n_v - rango, n_e - rango


vertices_triangulo = [0, 1, 2]
aristas_triangulo = [(0, 1), (1, 2), (2, 0)]
beta_0, beta_1 = betti_0_y_1(vertices_triangulo, aristas_triangulo)
grafo_nx = nx.Graph(aristas_triangulo)
assert beta_0 == 1 == nx.number_connected_components(grafo_nx)
assert beta_1 == 1 == len(nx.cycle_basis(grafo_nx))

vertices_dos_arboles = [0, 1, 2, 3, 4]
aristas_dos_arboles = [(0, 1), (1, 2), (3, 4)]
beta_0_b, beta_1_b = betti_0_y_1(vertices_dos_arboles, aristas_dos_arboles)
grafo_nx_2 = nx.Graph()
grafo_nx_2.add_nodes_from(vertices_dos_arboles)
grafo_nx_2.add_edges_from(aristas_dos_arboles)
assert beta_0_b == 2 == nx.number_connected_components(grafo_nx_2)
assert beta_1_b == 0 == len(nx.cycle_basis(grafo_nx_2))

print(f"Triangulo:   beta_0={beta_0}  beta_1={beta_1}")
print(f"Dos arboles: beta_0={beta_0_b}  beta_1={beta_1_b}")
```

Ejecutado, confirma `Triangulo: beta_0=1 beta_1=1` (un ciclo cerrado, un agujero) y `Dos arboles: beta_0=2 beta_1=0` (dos componentes, ningún ciclo) — coincidiendo exactamente con lo que `networkx` calcula por un camino algorítmico independiente.

### Conexión con sistemas de IA actuales

| Sistema | Componente topológica | Herramienta |
|---|---|---|
| Clasificación de imágenes | Fronteras de decisión | Homología persistente de activaciones |
| Detección de anomalías | Puntos fuera de la variedad de datos | Diagrama de persistencia |
| Análisis de robustez adversarial | Estabilidad de fronteras | Distancia de bottleneck $d_B$ |

---

## 🔄 El Ciclo del Agente

Los ocho capítulos anteriores dieron el lenguaje formal. Esta sección cierra la unidad aplicando ese lenguaje a una decisión de ingeniería real y recurrente al construir agentes: **¿en qué precisión numérica debe operar la memoria vectorial de un agente?** Un agente con memoria a largo plazo (Capítulo 10-bis de la fuente EMALCA: recuperación semántica sobre una base vectorial) suele comprimir sus embeddings con SVD de bajo rango (Capítulo 1) antes de indexarlos, y esa compresión puede hacerse en `float32` o en `float64`. La decisión no es cosmética: afecta memoria RAM, velocidad y precisión de recuperación en producción.

### Selección de Arquitectura

El agente necesita un mecanismo de compresión de memoria vectorial que reduzca el costo de almacenamiento sin degradar la calidad de recuperación semántica. La opción elegida es SVD truncada (Capítulo 1, fórmula 1.2): comprimir cada bloque de embeddings a rango $k\ll d$ y aceptar el error de reconstrucción acotado por el Teorema de Eckart–Young–Mirsky. La alternativa —no comprimir— escala linealmente en memoria con el número de documentos indexados y no es sostenible a partir de cientos de miles de vectores.

### Diseño

La pregunta de diseño concreta es la precisión numérica: ¿el `dtype` de los embeddings comprimidos debe ser `float32` o `float64`? La hipótesis de diseño es que, para este caso de uso, el error introducido por usar `float32` (precisión de máquina $\approx10^{-7}$) es despreciable frente al error que ya se acepta al truncar el rango — por lo que usar `float32` es una optimización de memoria sin costo de calidad perceptible.

### Implementación

```python
import numpy as np


def error_reconstruccion_svd(matriz: np.ndarray, rango: int, dtype: type) -> float:
    """Comprime `matriz` a rango `rango` via SVD truncada en el dtype dado
    y retorna el error de reconstruccion en norma de Frobenius (medido en
    float64 para comparar dtypes de forma justa)."""
    matriz_dtype = matriz.astype(dtype)
    U, S, Vt = np.linalg.svd(matriz_dtype, full_matrices=False)
    reconstruccion = (U[:, :rango] * S[:rango]) @ Vt[:rango, :]
    return float(
        np.linalg.norm(
            matriz.astype(np.float64) - reconstruccion.astype(np.float64), ord="fro"
        )
    )


np.random.seed(21)  # fija también el generador legado, por reproducibilidad explícita
rng = np.random.default_rng(21)
memoria_embeddings = rng.normal(size=(200, 64)).astype(np.float64)

error_f32 = error_reconstruccion_svd(memoria_embeddings, rango=8, dtype=np.float32)
error_f64 = error_reconstruccion_svd(memoria_embeddings, rango=8, dtype=np.float64)

print(f"Error de reconstruccion (float32): {error_f32}")
print(f"Error de reconstruccion (float64): {error_f64}")
print(f"Diferencia entre dtypes: {abs(error_f32 - error_f64):.2e}")
```

### Evaluación

```python
assert abs(error_f32 - error_f64) < 1e-3, "La diferencia por dtype debe ser despreciable"
assert error_f64 > 1.0, "El error dominante debe venir de truncar el rango, no del dtype"
```

Ejecutado, la diferencia entre `float32` y `float64` es del orden de `5e-8`, mientras que el error de compresión por truncar a rango 8 es del orden de `97` — cuatro órdenes de magnitud mayor. La hipótesis de diseño se confirma con evidencia numérica, no por intuición.

### Despliegue

Con esta evidencia, el agente indexa su memoria vectorial en `float32`: reduce a la mitad el uso de RAM y el ancho de banda de transferencia frente a `float64`, sin pérdida de calidad de recuperación medible frente al error que ya introduce la compresión de rango bajo.

### Iteración

Si en producción el rango de compresión $k$ se incrementa (por ejemplo, para mejorar la fidelidad de recuperación), el error de compresión decrece y eventualmente puede volverse comparable al error de `float32` — en ese punto, la decisión de diseño debe revisarse repitiendo esta misma medición con el nuevo $k$, no asumiendo que la conclusión de hoy sigue siendo válida indefinidamente.

### Diccionario de Variables

| Símbolo | Nombre | Descripción |
|---|---|---|
| `A` | Matriz de entrada | Matriz aleatoria $6\times4$ descompuesta por SVD (Cap. 1) |
| `U`, `S`, `Vt` | Factores de la SVD | $U\Sigma V^\top=A$; `S` son los valores singulares $\sigma_i$ (Cap. 1) |
| `rango` | Rango de truncamiento $k$ | Número de valores singulares conservados en `aproximar_rango_bajo` (Cap. 1) |
| `error_frobenius` | $\|A-A_k\|_F$ | Error de reconstrucción medido, comparado con la predicción teórica (Cap. 1) |
| `x`, `w1`, `w2` | Entrada y pesos de la red de 2 capas | Vectores de la red $z_1=\tanh(W_1x)$, $y=w_2\cdot z_1$ (Cap. 2) |
| `grad_w1_analitico` | $\partial L/\partial W_1$ | Gradiente calculado a mano vía la regla de la cadena (Cap. 2) |
| `grad_w1_numerico` | Gradiente por diferencias finitas | Verificación independiente del gradiente analítico (Cap. 2) |
| `r` | Parámetro de crecimiento del mapa logístico | Controla el régimen dinámico en $x_{n+1}=rx_n(1-x_n)$ (Cap. 3) |
| `lambda_estable`, `lambda_caotico` | Exponentes de Lyapunov $\lambda$ | Signo distingue estabilidad ($\lambda<0$) de caos ($\lambda>0$) (Cap. 3) |
| `X`, `y`, `theta_map` | Diseño, observaciones y estimador MAP | $\theta_{\text{MAP}}=(X^\top X+\lambda I)^{-1}X^\top y$ (Cap. 4) |
| `lambda_reg` | Coeficiente de regularización $\lambda$ | Precisión del prior gaussiano sobre $\theta$; equivale a weight decay $L_2$ (Cap. 4) |
| `p`, `q` | Distribuciones de probabilidad | Usadas para calcular entropía cruzada $H(p,q)$ y $D_{\text{KL}}(p\|q)$ (Cap. 5) |
| `phi` | Valores de Shapley $\phi_i$ | Contribución marginal promedio de cada jugador/feature (Cap. 5) |
| `dag_causal` | Grafo acíclico dirigido causal | Codifica relaciones causa→efecto entre `Confusor`, `Tratamiento` y `Resultado` (Cap. 5) |
| `Q`, `K`, `V` | Consultas, claves y valores | Matrices de entrada del mecanismo de atención (Cap. 6) |
| `matriz_atencion` (`A`) | Matriz de pesos de atención $\alpha_{ij}$ | $\text{softmax}(QK^\top/\sqrt{d_k})$; cada fila suma 1 (Cap. 6) |
| `d_k` | Dimensión de las claves | Factor de escala $\sqrt{d_k}$ en la atención escalada (Cap. 6) |
| `sigma`, `mu` | Desviación estándar y media | Parámetros de la gaussiana $N(\mu,\sigma^2)$ usada para la métrica de Fisher (Cap. 7) |
| `fisher_cerrada`, `fisher_montecarlo` | Información de Fisher $F(\mu)$ | Calculada en forma cerrada y verificada por Monte Carlo (Cap. 7) |
| `gradiente_natural` | Gradiente natural $F(\mu)^{-1}\nabla_\mu L$ | Reescala el gradiente euclidiano por el inverso de la métrica de Fisher (Cap. 7) |
| `vertices`, `aristas` | Vértices y aristas de un grafo | 0-símplices y 1-símplices del complejo simplicial (Cap. 8) |
| `boundary_1` | Operador frontera $\partial_1$ | Matriz cuyo rango determina $\beta_0$ y $\beta_1$ (Cap. 8) |
| `beta_0`, `beta_1` | Números de Betti $\beta_0$, $\beta_1$ | Componentes conexas y ciclos independientes del grafo (Cap. 8) |
| `error_f32`, `error_f64` | Error de reconstrucción por dtype | Comparación de precisión numérica en la compresión de memoria del agente (Ciclo del Agente) |

### Autoevaluación

```python
%%writefile test_unidad_0.py
import itertools
import math

import networkx as nx
import numpy as np
from scipy.optimize import minimize


def test_svd_reconstruye_la_matriz_y_cumple_eckart_young():
    """La SVD completa reconstruye A, y el error de rango bajo coincide
    con la prediccion del Teorema de Eckart-Young-Mirsky."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    A = rng.normal(size=(6, 4))
    U, S, Vt = np.linalg.svd(A)
    assert np.allclose(A, U[:, :4] @ np.diag(S) @ Vt)

    U2, S2, Vt2 = np.linalg.svd(A, full_matrices=False)
    A_2 = U2[:, :2] @ np.diag(S2[:2]) @ Vt2[:2, :]
    error_frobenius = np.linalg.norm(A - A_2, ord="fro")
    error_teorico = np.sqrt(np.sum(S[2:] ** 2))
    assert np.isclose(error_frobenius, error_teorico)


def test_backprop_manual_coincide_con_diferencias_finitas():
    """El gradiente analitico via regla de la cadena coincide con la
    diferenciacion numerica por diferencias finitas centradas."""
    np.random.seed(7)
    rng = np.random.default_rng(7)
    x = rng.normal(size=3)
    w1 = rng.normal(size=(2, 3))
    w2 = rng.normal(size=2)

    def perdida_escalar(x, w1, w2):
        z1 = np.tanh(w1 @ x)
        y = w2 @ z1
        return float(y**2)

    z1 = np.tanh(w1 @ x)
    y = w2 @ z1
    delta1 = (2.0 * y) * w2 * (1.0 - z1**2)
    grad_w1_analitico = np.outer(delta1, x)

    eps = 1e-6
    grad_w1_numerico = np.zeros_like(w1)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            w1_mas, w1_menos = w1.copy(), w1.copy()
            w1_mas[i, j] += eps
            w1_menos[i, j] -= eps
            grad_w1_numerico[i, j] = (
                perdida_escalar(x, w1_mas, w2) - perdida_escalar(x, w1_menos, w2)
            ) / (2 * eps)

    assert np.allclose(grad_w1_analitico, grad_w1_numerico, atol=1e-4)


def test_exponente_lyapunov_distingue_estabilidad_de_caos():
    """r=2.5 converge (lambda<0); r=3.9 es caotico (lambda>0)."""

    def exponente_lyapunov_logistico(r, n_iter, x0=0.4):
        x = x0
        suma_log = 0.0
        for _ in range(n_iter):
            suma_log += np.log(abs(r * (1.0 - 2.0 * x)))
            x = r * x * (1.0 - x)
        return suma_log / n_iter

    assert exponente_lyapunov_logistico(2.5, 5000) < 0
    assert exponente_lyapunov_logistico(3.9, 5000) > 0


def test_map_gaussiano_coincide_con_minimizacion_numerica():
    """theta_MAP en forma cerrada (ridge) coincide con minimizar el
    negativo log-posterior con un optimizador numerico independiente."""
    np.random.seed(3)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4))
    theta_verdadero = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ theta_verdadero + 0.1 * rng.normal(size=30)
    lambda_reg = 2.0

    theta_map = np.linalg.solve(X.T @ X + lambda_reg * np.eye(4), X.T @ y)

    def negativo_log_posterior(theta):
        residuo = y - X @ theta
        return float(residuo @ residuo + lambda_reg * (theta @ theta))

    resultado = minimize(negativo_log_posterior, x0=np.zeros(4), method="BFGS")
    assert np.allclose(theta_map, resultado.x, atol=1e-4)


def test_shapley_cumple_propiedad_de_eficiencia():
    """La suma de los valores de Shapley es exactamente v(N) - v(vacio)."""
    jugadores = ["A", "B", "C"]
    v = {
        frozenset(): 0.0, frozenset({"A"}): 1.0, frozenset({"B"}): 1.0,
        frozenset({"C"}): 0.0, frozenset({"A", "B"}): 4.0,
        frozenset({"A", "C"}): 2.0, frozenset({"B", "C"}): 2.0,
        frozenset({"A", "B", "C"}): 6.0,
    }
    n = len(jugadores)
    phi = {j: 0.0 for j in jugadores}
    for jugador in jugadores:
        resto = [j for j in jugadores if j != jugador]
        for tam in range(len(resto) + 1):
            for subconjunto in itertools.combinations(resto, tam):
                s = frozenset(subconjunto)
                peso = (
                    math.factorial(len(s)) * math.factorial(n - len(s) - 1)
                ) / math.factorial(n)
                phi[jugador] += peso * (v[s | {jugador}] - v[s])

    assert math.isclose(sum(phi.values()), v[frozenset(jugadores)] - v[frozenset()])


def test_dag_causal_es_aciclico_y_tiene_confusor():
    """El DAG causal de ejemplo es aciclico y 'Resultado' tiene como
    padres tanto al Confusor como al Tratamiento."""
    dag = nx.DiGraph()
    dag.add_edges_from(
        [("Confusor", "Tratamiento"), ("Confusor", "Resultado"), ("Tratamiento", "Resultado")]
    )
    assert nx.is_directed_acyclic_graph(dag)
    assert set(dag.predecessors("Resultado")) == {"Confusor", "Tratamiento"}


def test_atencion_escalada_es_distribucion_y_combinacion_convexa():
    """Cada fila de la matriz de atencion suma 1 y la salida es una
    combinacion convexa de V."""

    def softmax(z):
        z_estable = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_estable)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    np.random.seed(11)
    rng = np.random.default_rng(11)
    Q = rng.normal(size=(4, 8))
    K = rng.normal(size=(4, 8))
    V = rng.normal(size=(4, 6))
    A = softmax((Q @ K.T) / np.sqrt(8))
    salida = A @ V

    assert np.allclose(A.sum(axis=1), 1.0)
    for i in range(4):
        assert np.allclose(sum(A[i, j] * V[j] for j in range(4)), salida[i], atol=1e-10)


def test_gradiente_natural_reescala_por_inverso_de_fisher():
    """Para N(mu, sigma^2) con sigma=2, F(mu)=0.25 y el gradiente natural
    debe ser 4 veces el gradiente euclidiano."""
    sigma = 2.0
    fisher = 1.0 / sigma**2
    gradiente_euclidiano = 0.0 - 5.0
    gradiente_natural = (1.0 / fisher) * gradiente_euclidiano
    assert np.isclose(gradiente_natural, 4.0 * gradiente_euclidiano)


def test_numeros_de_betti_coinciden_con_networkx():
    """beta_0 y beta_1 calculados via el operador frontera coinciden con
    los calculados por networkx (componentes conexas y base de ciclos)."""

    def betti_0_y_1(vertices, aristas):
        n_v, n_e = len(vertices), len(aristas)
        indice = {v: i for i, v in enumerate(vertices)}
        boundary_1 = np.zeros((n_v, n_e))
        for j, (a, b) in enumerate(aristas):
            boundary_1[indice[a], j] = -1.0
            boundary_1[indice[b], j] = 1.0
        rango = np.linalg.matrix_rank(boundary_1)
        return n_v - rango, n_e - rango

    vertices, aristas = [0, 1, 2], [(0, 1), (1, 2), (2, 0)]
    beta_0, beta_1 = betti_0_y_1(vertices, aristas)
    grafo = nx.Graph(aristas)
    assert beta_0 == 1 == nx.number_connected_components(grafo)
    assert beta_1 == 1 == len(nx.cycle_basis(grafo))


def test_error_float32_despreciable_frente_a_error_de_compresion():
    """La diferencia de error de reconstruccion SVD entre float32 y
    float64 es ordenes de magnitud menor que el error de truncar el rango."""

    def error_reconstruccion_svd(matriz, rango, dtype):
        m = matriz.astype(dtype)
        U, S, Vt = np.linalg.svd(m, full_matrices=False)
        reconstruccion = (U[:, :rango] * S[:rango]) @ Vt[:rango, :]
        return float(
            np.linalg.norm(
                matriz.astype(np.float64) - reconstruccion.astype(np.float64), ord="fro"
            )
        )

    np.random.seed(21)
    rng = np.random.default_rng(21)
    memoria = rng.normal(size=(200, 64)).astype(np.float64)
    error_f32 = error_reconstruccion_svd(memoria, 8, np.float32)
    error_f64 = error_reconstruccion_svd(memoria, 8, np.float64)
    assert abs(error_f32 - error_f64) < 1e-3
    assert error_f64 > 1.0
```

```python
!pytest test_unidad_0.py -v
```
