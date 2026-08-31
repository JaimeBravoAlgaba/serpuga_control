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

El signo de \(v_i\) es físico: valores positivos y negativos representan el
avance o retroceso de cada banda sobre su propio eje longitudinal.

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
3. Seguimiento de posición, orientación, velocidad y velocidad angular.
4. Deslizamiento, *scrubbing* y suavidad de los comandos.

Los parámetros geométricos incluidos son ilustrativos. Antes de pasar al robot
real deben sustituirse las posiciones de pivotes, offsets, masas, altura del
centro de masas y límites de los actuadores.

La configuración nominal y el acoplamiento de simetría son también parámetros
del robot. Esto permite describir tanto el montaje paralelo, con
\(q_1+q_2=0\), como el montaje antiparalelo, con \(q_2-q_1=\pi\), sin cambiar
la formulación del controlador.

## Configuración y ejecución online

Los parámetros de una ejecución se agrupan en un único perfil YAML con cuatro
secciones: `robot`, `scenario`, `simulation` y `mpc`. Las magnitudes angulares
de geometría se expresan en grados en el archivo y se convierten internamente a
radianes. La referencia se define mediante una velocidad lineal y una velocidad
angular constantes, integradas desde el `x,y` inicial y con orientación de
referencia inicial alineada con el corredor.

La aplicación gráfica utiliza una sesión de bucle cerrado persistente. Cada
llamada resuelve solamente el horizonte correspondiente al instante actual,
aplica el primer comando, integra un periodo y publica inmediatamente el nuevo
estado al visualizador. El historial se conserva para las gráficas y la
exportación, pero no se calcula antes de comenzar la animación.

No se configura una postura especial de entrada al hueco. El plegado de las
orugas es una decisión del NMPC causada por las restricciones geométricas,
estabilidad, deslizamiento, límites articulares y costes de actuación. La
semilla numérica del solver solo usa la anchura disponible y los límites
articulares para encontrar un punto inicial factible cuando la configuración
nominal no cabe.

El yaw inicial pertenece al estado del robot. No rota la referencia de posición:
la referencia traslacional empieza en el mismo `x,y` y avanza por el eje del
corredor. El umbral de orientación del MPC es blando, de forma que una
orientación inicial distinta penaliza el objetivo pero no hace infeasible el
primer paso.
