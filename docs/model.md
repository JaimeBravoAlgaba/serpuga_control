# Modelo del prototipo

## Variables

La primera versión utiliza el estado

\[
x=[X,\,Y,\,\psi,\,q_1,\,q_2]^T
\]

y los comandos

\[
u=[v_1,\,v_2,\,\dot q_1,\,\dot q_2]^T.
\]

El *twist* plano del cuerpo

\[
\xi=[v_x,\,v_y,\,\omega]^T
\]

no es una entrada independiente ni una variable libre del NMPC. Se obtiene de
los comandos de las orugas mediante la proyección cinemática de mínimo
deslizamiento descrita a continuación. Por tanto, el optimizador no puede
ordenar directamente una velocidad lateral del robot.

## Deslizamiento

Para la oruga `i`, el centro de la huella respecto al cuerpo es

\[
r_i(q_i)=p_i+R(q_i)\rho_i.
\]

Su velocidad es

\[
u_{c,i}=v_B+\omega J r_i+\frac{\partial r_i}{\partial q_i}\dot q_i,
\]

y el residuo de deslizamiento

\[
s_i=u_{c,i}-v_i e_i.
\]

La velocidad impuesta por cada banda es siempre longitudinal:

\[
v_i e_i(q_i).
\]

El twist utilizado en la dinámica se calcula como

\[
\xi(q,u)=\arg\min_\xi
\sum_{i=1}^{2}
\left(w_\parallel s_{i,\parallel}^2+
w_\perp s_{i,\perp}^2\right)+\varepsilon\lVert\xi\rVert^2.
\]

Así, el deslizamiento lateral puede aparecer pasivamente por incompatibilidad
geométrica o durante la reconfiguración, pero no actúa como un grado de libertad
controlable. El NMPC lo penaliza y puede imponer además una cota máxima. En el
escenario antiparalelo dicha cota es 0,02 m/s.

Se añade también un término de *scrubbing* proporcional a
`(omega + q_dot_i)^2` para representar el giro de una huella finita.

## Corredor

Cada vértice de ambas huellas y del conector rígido debe satisfacer

\[
y_\mathrm{lower}(x)+\delta \le y_j \le
y_\mathrm{upper}(x)-\delta.
\]

`StraightGapCorridor` simula la salida del futuro bloque de percepción. La
transición entre la zona abierta y el hueco utiliza funciones hiperbólicas
suaves para mantener derivables las restricciones.

## Estabilidad

El soporte lateral se obtiene proyectando las esquinas de las dos huellas sobre
la normal de la trayectoria. Si está activado el ZMP, el punto evaluado es

\[
p_\mathrm{ZMP}=c_{xy}-\frac{h}{g}a_{xy}.
\]

Se exige un margen mínimo respecto a ambos extremos del intervalo de soporte y
se penalizan márgenes inferiores al objetivo. Esta es todavía una aproximación
de suelo plano y altura constante, no un modelo dinámico de vuelco.

## Prioridades

1. Geometría del corredor y límites articulares.
2. Margen mínimo de estabilidad.
3. Seguimiento de pose, velocidad y velocidad angular.
4. Deslizamiento, *scrubbing* y suavidad de los comandos.

Los parámetros geométricos incluidos son ilustrativos. Antes de pasar al robot
real deben sustituirse las posiciones de pivotes, offsets, masas, altura del
centro de masas y límites de los actuadores.

La configuración nominal y el acoplamiento de simetría son también parámetros
del robot. Esto permite describir tanto el montaje paralelo, con
\(q_1+q_2=0\), como el montaje antiparalelo, con \(q_2-q_1=\pi\), sin cambiar
la formulación del controlador.
