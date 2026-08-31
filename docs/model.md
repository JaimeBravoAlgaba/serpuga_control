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

El MPC limita esta velocidad y su variación. Esta aproximación permite separar
la cinemática inversa del futuro modelo dinámico de los actuadores.

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
modelo puede reflejar deslizamiento transitorio durante la reorientación y
*scrubbing* debido a la longitud finita de la huella. El MPC penaliza las
componentes longitudinal y lateral con pesos distintos y puede imponer una
cota máxima a la componente lateral.

## Formulación del MPC

En cada instante del horizonte, el optimizador decide solamente

\[
u_k=[v_{x,k},\,v_{y,k},\,\omega_k]^T.
\]

Dentro del grafo simbólico de CasADi se evalúa la cinemática inversa para
obtener \(q_{i,k}^*\), \(v_{i,k}^*\) y \(\dot q_{i,k}\). Sobre estas magnitudes
se aplican:

- límites de velocidad y aceleración de las bandas;
- límites de posición, velocidad y aceleración articular;
- cota de velocidad lineal y angular de la barra;
- restricciones de deslizamiento lateral;
- restricciones geométricas del corredor para todos los vértices;
- margen mínimo de estabilidad geométrica o basado en ZMP.

El coste conserva el seguimiento de posición, orientación, velocidad lineal y
angular, junto con deslizamiento, *scrubbing*, esfuerzo de bandas, velocidad
articular, suavidad del *twist*, configuración nominal y estabilidad.

Una consecuencia importante es que la forma del robot ya no puede elegirse de
manera independiente: debe ser compatible con el movimiento instantáneo de la
barra. Para estrecharse, el MPC busca una secuencia de \(v_x\), \(v_y\) y
\(\omega\) cuya cinemática inversa produzca configuraciones que entren en el
hueco y respeten simultáneamente el seguimiento y la estabilidad.

## Corredor y estabilidad

Cada vértice de ambas huellas y del conector debe satisfacer

\[
y_{lower}(x)+\delta \le y_j \le y_{upper}(x)-\delta.
\]

`StraightGapCorridor` simula la salida del futuro bloque de percepción. Para
la estabilidad, el soporte lateral se obtiene proyectando las esquinas de las
orugas sobre la normal de la trayectoria. Cuando se activa la aproximación ZMP,
el punto evaluado es

\[
p_{ZMP}=c_{xy}-\frac{h}{g}a_{xy}.
\]

Se exige un margen mínimo respecto a ambos extremos del intervalo de soporte.
Sigue siendo una aproximación bidimensional sobre suelo plano, no un modelo
dinámico completo de vuelco.

## Simulación y teleoperación

El visualizador conserva por separado:

- `controls`: \([v_x,v_y,\omega]\), las decisiones del MPC o del usuario;
- `actuator_commands`: \([q_1,q_2,v_1,v_2]\), la salida de la cinemática
  inversa.

En modo manual, los tres deslizadores controlan directamente \(v_x\), \(v_y\)
y \(\omega\). La misma cinemática inversa que usa el MPC genera las consignas
de las orugas en vivo.
