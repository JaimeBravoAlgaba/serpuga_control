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

es una variable algebraica del NMPC. Esto convierte el problema en una
cinemática inversa predictiva: el controlador selecciona simultáneamente el
movimiento rígido y los comandos de las orugas, penalizando la incompatibilidad
entre ambos mediante el deslizamiento.

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

El coste separa las componentes longitudinal y lateral, dando mayor peso a la
segunda. Se añade además un término de *scrubbing* proporcional a
`(omega + q_dot_i)^2` para representar el giro de una huella finita.

`KinematicModel.body_twist()` conserva también una proyección directa de mínimo
deslizamiento. Es útil como modelo auxiliar y para comprobar comandos, pero el
NMPC usa la formulación inversa anterior para no confundir «mínimo slip» con
«robot detenido».

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

