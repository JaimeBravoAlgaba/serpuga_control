# Modelo cinemático y control

## Estado y variables de control

El estado de predicción es

\[
x=[X,\,Y,\,\psi,\,q_1,\,q_2]^T,
\]

donde \((X,Y)\) es el centro de la barra rígida, \(\psi\) su orientación y
\(q_i\) la orientación relativa de cada oruga.

Las únicas variables de control del MPC son el *twist* plano del centro de la
barra, expresado en el sistema ligado a ella:

\[
u=\xi_B=[v_x,\,v_y,\,\omega]^T.
\]

Por tanto, `v1`, `v2`, `q1` y `q2` ya no son variables independientes del
optimizador. Se calculan mediante la cinemática inversa analítica.

## Cinemática inversa analítica

Sea \(p_i\) la posición del pivote de la oruga \(i\) respecto al centro de la
barra y

\[
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
\]

La velocidad requerida en cada pivote es

\[
u_i=
\begin{bmatrix}v_x\\v_y\end{bmatrix}
+\omega Jp_i.
\]

Como una oruga solo puede generar velocidad longitudinal, su eje debe ser
paralelo a \(u_i\). Puesto que la velocidad de banda es firmada, las
orientaciones \(q_i\) y \(q_i+\pi\) representan el mismo eje físico. Se elige
la solución equivalente más próxima al ángulo actual para evitar giros
innecesarios de 180 grados.

La implementación emplea la siguiente forma analítica y regularizada. Con

\[
a_i=e(q_i)^Tu_i,
\qquad
b_i=n(q_i)^Tu_i,
\]

se calcula

\[
\Delta q_i=
\frac{1}{2}
\operatorname{atan2}
\left(
2a_ib_i,
a_i^2-b_i^2+\varepsilon^2
\right),
\]

\[
q_i^*=q_i+\Delta q_i,
\qquad
v_i^*=e(q_i^*)^Tu_i.
\]

La salida de la inversa tiene el orden

\[
u_{act}=[q_1^*,\,q_2^*,\,v_1^*,\,v_2^*]^T.
\]

Si \(u_i=0\), se conserva el ángulo anterior y se asigna \(v_i^*=0\). El
signo de \(v_i^*\) es físico: permite que una oruga montada a 180 grados se
desplace en sentido opuesto sin reorientarse.

En el contacto puntual ideal del pivote se cumple

\[
v_i^*e(q_i^*)=u_i,
\]

por lo que la solución tiene deslizamiento nulo en ese punto siempre que los
límites articulares y de actuación permitan aplicarla.

## Evolución del estado

La pose de la barra se integra directamente desde el control:

\[
\begin{bmatrix}\dot X\\\dot Y\end{bmatrix}
=R(\psi)
\begin{bmatrix}v_x\\v_y\end{bmatrix},
\qquad
\dot\psi=\omega.
\]

La implementación usa RK4 con control constante durante cada periodo. Para las
articulaciones se adopta inicialmente un servo ideal de posición: al final del
periodo se alcanza \(q_i^*\). La velocidad articular media requerida queda
explícita:

\[
\dot q_i=\frac{q_i^*-q_i}{\Delta t}.
\]

Esta velocidad queda disponible como diagnóstico, pero no se limita en la
formulación simplificada. La aproximación permite separar la cinemática inversa
del futuro modelo dinámico de los actuadores.

## Deslizamiento de una huella finita

La inversa anterior usa los pivotes como contactos puntuales. Para conservar
la información geométrica de una oruga real, el coste y las restricciones de
deslizamiento se evalúan en el centro de cada huella.

Si

\[
r_i(q_i)=p_i+R(q_i)\rho_i,
\]

su velocidad es

\[
u_{c,i}=v_B+\omega Jr_i+
\frac{\partial r_i}{\partial q_i}\dot q_i,
\]

y el residuo es

\[
s_i=u_{c,i}-v_i^*e(q_i).
\]

Así, el contacto puntual de la inversa puede ser exacto y, al mismo tiempo, el
simulador puede mostrar deslizamiento transitorio durante la reorientación y
*scrubbing* debido a la longitud finita de la huella. Estas magnitudes se
registran como diagnósticos: no forman parte del coste ni de las restricciones
del MPC simplificado.

## Formulación del MPC

En cada instante del horizonte, el optimizador decide solamente

\[
u_k=[v_{x,k},\,v_{y,k},\,\omega_k]^T.
\]

Dentro del grafo simbólico de CasADi se evalúa la cinemática inversa para
obtener \(q_{i,k}^*\) y \(v_{i,k}^*\). El coste de etapa contiene únicamente
seguimiento y paralelismo:

\[
\begin{aligned}
\ell_k={}&w_p\|p_k-p_k^r\|^2
+2w_\psi[1-\cos(\psi_k-\psi_k^r)]\\
&+w_v\|v_k^W-v_k^r\|^2
+w_\omega(\omega_k-\omega_k^r)^2\\
&+w_{\parallel}\sin^2(q_{1,k}-q_{2,k}).
\end{aligned}
\]

El último término vale cero tanto si las orugas apuntan en el mismo sentido
como si sus direcciones difieren \(180^\circ\). Por tanto mide el paralelismo
de sus ejes físicos, no el signo de las velocidades de banda. Se añaden los
términos terminales de posición y orientación de la misma referencia.

La formulación conserva solo cuatro familias de desigualdades. La condición
inicial y la dinámica se imponen mediante disparo múltiple:

\[
x_0=x_{medido},\qquad x_{k+1}=f(x_k,u_k),
\]

\[
q_{min}\le q_k\le q_{max},
\qquad
\|[v_x,v_y]^T\|_2\le v_{max},
\qquad
|\omega|\le\omega_{max},
\]

\[
W_{robot}(x_k,n_k)\le W_{libre}(X_k)-2\delta.
\]

No hay cotas de deslizamiento, ZMP, velocidad o aceleración articular, bandas,
*scrubbing*, simetría ni restricciones independientes por vértice.

## Restricción única de anchura

Sea \(n_k=[-\sin\psi_k^r,\cos\psi_k^r]^T\) la normal a la trayectoria. La
anchura centrada que necesita la formación, incluyendo ambas orugas, el brazo
y cualquier error lateral respecto al centro de la trayectoria, es

\[
W_{robot}=2\max_{z_j\in\mathcal F}
\left|n_k^T(z_j-p_k^r)\right|,
\]

donde \(\mathcal F\) contiene los doce vértices de la huella geométrica. Para
anticipar una transición, el ancho disponible es el menor valor del corredor
bajo toda la huella longitudinal:

\[
W_{libre}=\min_{z_j\in\mathcal F}W_{corredor}(z_{j,x}).
\]

En el grafo simbólico se emplean máximos y mínimos suavizados con
\(\varepsilon=10^{-3}\). `StraightGapCorridor` simula el valor que más adelante
proporcionará el bloque láser. Aunque se evalúen los vértices para calcular
ambos escalares, el optimizador recibe una sola desigualdad de anchura por
instante, no una restricción independiente por vértice.

La referencia \(p_k^r\) hace que la misma magnitud incluya el centrado lateral.
El margen de soporte geométrico, la holgura real vértice-pared y el
deslizamiento se siguen calculando solo para diagnóstico y validación posterior.

Si IPOPT consume su presupuesto de tiempo, el controlador no aplica una
iteración arbitraria: conserva o reconstruye una secuencia y comprueba
numéricamente dinámica, articulaciones, módulo de velocidad, velocidad angular
y anchura antes de usar su primer mando. Esta reserva no introduce un objetivo
nuevo; solo mantiene una salida factible en ejecución online.

Eliminar velocidades articulares y deslizamiento es deliberado, pero tiene una
consecuencia visible: el servo ideal puede pedir cambios grandes de \(q_i\) y el
diagnóstico de *slip* de huella puede crecer. Debe interpretarse como una
limitación de esta versión mínima, no como una predicción apta para hardware.

## Simulación y teleoperación

El visualizador conserva por separado:

- `controls`: \([v_x,v_y,\omega]\), las decisiones del MPC o del usuario;
- `actuator_commands`: \([q_1,q_2,v_1,v_2]\), la salida de la cinemática
  inversa.

En modo manual, los tres deslizadores controlan directamente \(v_x\), \(v_y\)
y \(\omega\). La misma cinemática inversa que usa el MPC genera las consignas
de las orugas en vivo.
