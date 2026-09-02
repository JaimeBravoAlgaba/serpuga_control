# Modelo cinemático y control

## Estado y variables de control

El estado de predicción es

\[
x=[X,\,Y,\,\psi,\,q_1,\,q_2]^T,
\]

donde \((X,Y)\) es el centro de la barra rígida, \(\psi\) su orientación y
\(q_i\) la orientación relativa instantánea de cada oruga.

Las variables de control del NMPC son directamente las consignas de actuación:

\[
u=[q_{1,cmd},\,q_{2,cmd},\,v_1,\,v_2]^T.
\]

Por tanto, no existe una etapa de cinemática inversa entre el optimizador y los
actuadores. Las dos primeras componentes son objetivos articulares al final del
periodo de control y las dos últimas son velocidades longitudinales firmadas de
las bandas.

## Cinemática directa desde las orugas

Sea \(p_i=[p_{x,i},p_{y,i}]^T\) la posición del pivote de la oruga \(i\) respecto
al centro de la barra y sea

\[
e(q_i)=\begin{bmatrix}\cos q_i\\\sin q_i\end{bmatrix}.
\]

Para una velocidad rígida de la barra

\[
\xi_B=[v_x,\,v_y,\,\omega]^T,
\]

la velocidad del pivote \(i\) es

\[
u_i=
\begin{bmatrix}v_x\\v_y\end{bmatrix}
+\omega
\begin{bmatrix}-p_{y,i}\\p_{x,i}\end{bmatrix}.
\]

La oruga solicita en ese punto la velocidad longitudinal

\[
u_i^d=v_i e(q_i).
\]

Apilando las dos ecuaciones vectoriales se obtiene un sistema de cuatro
ecuaciones para tres componentes de twist,

\[
A\xi_B=b(q_1,q_2,v_1,v_2).
\]

Como ambas orugas pueden pedir velocidades incompatibles con un único movimiento
rígido, el modelo calcula la solución regularizada

\[
\xi_B=(A^TA+\lambda I)^{-1}A^Tb.
\]

El pequeño término de Tikhonov \(\lambda\) está configurado por `regularisation`.
El twist resultante es una variable derivada: sirve para integrar la pose,
calcular el seguimiento y diagnosticar incompatibilidades, pero no es una salida
directa del MPC.

## Movimiento articular dentro del periodo

La restricción de velocidad articular se define como

\[
\dot q_i=\frac{q_{i,cmd}-q_i}{\Delta t},
\qquad
|\dot q_i|\le \dot q_{max}.
\]

A diferencia de una aproximación de servo instantáneo, el modelo no utiliza
`q_cmd` durante todo el intervalo. Se asume una interpolación lineal:

\[
q_i(\alpha)=q_i(k)+\alpha\,[q_{i,cmd}(k)-q_i(k)],
\qquad \alpha\in[0,1].
\]

La integración RK4 de la pose evalúa la cinemática con los ángulos de inicio,
mitad y final del intervalo. Al terminar el periodo se cumple

\[
q_i(k+1)=q_{i,cmd}(k).
\]

Así, limitar la velocidad articular tiene efecto también sobre la predicción de
la pose: una reconfiguración no modifica instantáneamente la dirección efectiva
de las orugas.

## Evolución de la pose

Para cada valor instantáneo de \(q\), la cinemática directa proporciona
\(\xi_B(q,v_1,v_2)\). Entonces

\[
\begin{bmatrix}\dot X\\\dot Y\end{bmatrix}
=R(\psi)
\begin{bmatrix}v_x\\v_y\end{bmatrix},
\qquad
\dot\psi=\omega.
\]

La velocidad del mundo utilizada en el coste de seguimiento es la velocidad
media del paso discreto,

\[
v_k^W=\frac{p_{k+1}-p_k}{\Delta t},
\]

por lo que incorpora la evolución articular durante el intervalo.

## Deslizamiento e incompatibilidad cinemática

Para unos ángulos reales \(q_i\), el residuo en los pivotes es

\[
r=A\xi_B-b.
\]

Cada bloque \(r_i\in\mathbb R^2\) se proyecta sobre la dirección longitudinal y
lateral de la oruga:

\[
s_{i,\parallel}=e(q_i)^T r_i,
\qquad
s_{i,\perp}=n(q_i)^T r_i,
\]

con

\[
n(q_i)=\begin{bmatrix}-\sin q_i\\\cos q_i\end{bmatrix}.
\]

Estas magnitudes se registran como diagnóstico. No forman parte del coste ni de
las restricciones del NMPC simplificado.

## Formulación del NMPC

En cada etapa el optimizador decide

\[
u_k=[q_{1,cmd,k},\,q_{2,cmd,k},\,v_{1,k},\,v_{2,k}]^T.
\]

La dinámica se impone mediante disparo múltiple:

\[
x_0=x_{medido},
\qquad
x_{k+1}=f(x_k,u_k).
\]

El coste de etapa es

\[
\begin{aligned}
\ell_k={}&w_p\|p_k-p_k^r\|^2
+2w_\psi[1-\cos(\psi_k-\psi_k^r)]\\
&+w_v\|v_k^W-v_k^r\|^2
+w_\omega(\omega_{mid,k}-\omega_k^r)^2\\
&+w_{\parallel}\sin^2(q_{1,mid,k}-q_{2,mid,k}).
\end{aligned}
\]

El término de paralelismo vale cero tanto para ejes paralelos como antiparalelos.
Se añaden términos terminales de posición y orientación.

## Restricciones de actuación y velocidad

Se imponen

\[
q_{min}\le q_k\le q_{max},
\qquad
q_{min}\le q_{cmd,k}\le q_{max},
\]

\[
|q_{cmd,k}-q_k|\le \dot q_{max}\Delta t,
\]

\[
|v_{i,k}|\le v_{track,max}.
\]

Además, el twist derivado debe cumplir

\[
\|[v_x,v_y]^T\|_2\le v_{body,max},
\qquad
|\omega|\le\omega_{max}.
\]

Estas dos últimas cotas se comprueban en los ángulos articulares de inicio,
mitad y final de cada periodo, no únicamente en `q_cmd`.

## Restricción geométrica del corredor

`StraightGapCorridor` describe un corredor cuyo centro lateral es `centre_y` y
cuya anchura depende de la coordenada longitudinal \(x\). Sus paredes son

\[
y_{sup}(x)=y_c+\frac{W(x)}{2},
\qquad
y_{inf}(x)=y_c-\frac{W(x)}{2}.
\]

Para cada vértice mundial \(z_j=[x_j,y_j]^T\) de la huella completa se define

\[
g_j=|y_j-y_c|+\delta-\frac{W(x_j)}{2},
\]

donde \(\delta\) es `clearance_margin`.

La condición física de no colisión es \(g_j\le 0\) para todos los vértices. Para
mantener una única desigualdad geométrica por estado, el controlador construye

\[
g_{corr}=\operatorname{smax}_\varepsilon(g_1,\ldots,g_m)\le 0,
\]

usando valor absoluto y máximo suavizados. Esta formulación respeta directamente
las paredes del corredor y su desplazamiento `centre_y`; no utiliza la posición
de la referencia como centro geométrico del hueco.

La restricción se evalúa en los estados discretos y también en el estado
intermedio de cada periodo para reducir el riesgo de una colisión transitoria
durante la reconfiguración.

## Reserva factible

Si IPOPT no devuelve una solución dentro del presupuesto configurado, el
controlador comprueba numéricamente la secuencia de warm-start antes de usarla.
Se verifican:

- límites articulares y de velocidad articular;
- límites de velocidad de banda;
- velocidad lineal y angular derivadas al inicio, mitad y final de cada paso;
- despeje frente al corredor en estados y puntos intermedios.

Sólo una secuencia que supera todas estas comprobaciones puede utilizarse como
fallback.

## Simulación y teleoperación

`SimulationLog.controls` y `SimulationLog.actuator_commands` contienen el mismo
orden de cuatro consignas:

\[
[q_{1,cmd},q_{2,cmd},v_1,v_2].
\]

`body_twists` almacena por separado el twist derivado de cada intervalo y
`world_velocities` la velocidad media del mundo utilizada para el seguimiento.
Los visualizadores muestran explícitamente ambas familias de magnitudes para no
confundir ángulos articulares con velocidades cartesianas.

El modo manual conserva los tres mandos `[vx, vy, omega]` únicamente como capa de
compatibilidad de interfaz. En ese modo las articulaciones permanecen en su
posición actual y las velocidades de banda se proyectan sobre los ejes reales de
las orugas; el NMPC no interviene.

## Alcance

El modelo es todavía cinemático. No incluye aceleraciones, pares, dinámica
interna de actuadores, fuerzas de contacto ni dinámica completa del terreno. La
interpolación articular y las restricciones intermedias corrigen la inconsistencia
de considerar `q_cmd` instantáneo, pero no sustituyen un modelo dinámico de bajo
nivel para ejecución sobre hardware.
