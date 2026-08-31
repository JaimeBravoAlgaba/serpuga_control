# SERPUGA Control

Prototipo en Python de control predictivo no lineal para un robot formado por
dos orugas reconfigurables. El controlador sigue una trayectoria con referencias
de velocidad lineal y angular, minimiza el deslizamiento, adapta la envolvente
del robot a un corredor estrecho y conserva un margen lateral de estabilidad.

## Qué incluye

- descripción paramétrica de pivotes, huellas, masas y límites de actuación;
- cinemática plana con contribución de `q_dot` cuando el pivote no coincide con
  el centro de la huella;
- cinemática predictiva con `v1`, `v2`, `q1_dot` y `q2_dot`, sin un twist
  omnidireccional independiente;
- velocidades de banda firmadas, permitiendo avance y retroceso por oruga;
- deslizamiento longitudinal, lateral y término de *scrubbing*;
- NMPC de disparo múltiple implementado con CasADi/IPOPT;
- restricciones sobre todos los vértices del robot;
- corredor sintético intercambiable por el futuro estimador láser;
- estabilidad lateral geométrica o mediante ZMP aproximado;
- aplicación gráfica con configuración completa y perfiles YAML;
- simulación online: cada paso MPC se resuelve, aplica y dibuja inmediatamente;
- teleoperación manual en vivo con consignas articulares y velocidades de oruga;
- carga de parámetros, pausa, reanudación y parada sin precalcular la trayectoria
  ejecutada.

## Estructura

| Módulo | Función |
|---|---|
| `config.py` | Dataclasses numéricas consumidas por el controlador |
| `configuration.py` | Esquema de interfaz, validación y perfiles YAML |
| `robot.py` | Geometría, huellas, CoM y soporte |
| `kinematics.py` | Twist, slip e integración RK4 |
| `trajectory.py` | Referencias y previsualización |
| `corridor.py` | Estimación sintética del espacio libre |
| `nmpc.py` | Formulación y resolución del NMPC |
| `runtime.py` | Construcción del modelo y controlador desde un perfil |
| `simulation.py` | Sesión online paso a paso, bucle batch y métricas |
| `app.py` | Aplicación de escritorio y editor de parámetros |
| `online_visualization.py` | Visualización actualizada durante la optimización |
| `playback.py` | Estado de reproducción y navegación temporal |
| `live_visualization.py` | Reproductor histórico para exportar vídeos |
| `visualization.py` | Generador opcional de informes estáticos |
| `cli.py` | Ejecución reproducible de escenarios |

La formulación matemática se resume en [`docs/model.md`](docs/model.md).

## Instalación

Con Conda, desde la raíz del repositorio:

```bash
conda env create -f environment.yml
conda activate serpuga-control
```

El entorno incluye FFmpeg para poder exportar las reproducciones en MP4 y
deja el paquete instalado en modo editable.

Como alternativa, con `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Aplicación interactiva

Desde la raíz del repositorio:

```bash
python -m serpuga_control
```

La ventana se abre con el robot, el corredor y la simulación online ya
inicializados. A la derecha aparecen cuatro pestañas:

- **Robot**: geometría, masas, articulaciones, actuadores y pesos de contacto;
- **Escenario**: anchuras, posición y transición del hueco;
- **Simulación**: estado inicial, duración y referencias constantes `v` y `ω`;
- **MPC**: horizonte, costes, restricciones, ZMP y opciones de IPOPT.

La simulación corre en tiempo real desde que se carga un perfil. El panel de
**Teleoperación** permite activar **Modo manual**, que deja de resolver el MPC y
aplica directamente las velocidades `v1`, `v2` y las consignas articulares
`q1`, `q2` en el siguiente periodo de control. En manual, la simulación no se
detiene por `duration` ni por `stop_x`.

Al pulsar **Cargar parámetros**, la aplicación reconstruye la sesión con los
valores visibles en las pestañas. Después resuelve una única iteración, aplica
ese control al modelo y actualiza la figura antes de pasar a la siguiente. No
existe un cálculo previo de toda la ejecución. Si una iteración tarda más que el
periodo configurado, la barra inferior muestra el ritmo real alcanzado en lugar
de ocultar el retraso.

## Configuraciones YAML

Los perfiles se guardan en [`configs/`](configs/). El desplegable superior lista
automáticamente todos los archivos `.yaml` y `.yml` de esa carpeta. **Cargar**
rellena todas las pestañas y **Guardar como…** crea o actualiza un perfil con los
valores actuales.

Se incluyen tres ejemplos editables:

- `default.yaml`: orugas antiparalelas y hueco de 0,58 m;
- `parallel-gap.yaml`: partida con las orugas paralelas;
- `open-turn.yaml`: corredor abierto y referencia angular no nula.

También puede elegirse el perfil inicial al abrir la interfaz:

```bash
python -m serpuga_control --config parallel-gap
```

Para listar perfiles o ejecutar uno sin interfaz:

```bash
python -m serpuga_control --list-configs
python -m serpuga_control --config default --headless
```

La exportación de capturas y vídeos se mantiene como operación batch:

```bash
python -m serpuga_control --config default --screenshot artifacts/gap.png
python -m serpuga_control --config default --video artifacts/run.mp4
```

También se instala el comando equivalente `serpuga-demo`.

## Pruebas

```bash
pytest
```

## Resultado de referencia

Con los parámetros de demostración, el robot empieza a plegarse antes del
hueco, mantiene 10 mm de separación mínima respecto al corredor y recupera la
configuración paralela tras salir. Los resultados exactos dependen del equipo y
se imprimen como JSON al terminar cada ejecución.

## Alcance actual

Este repositorio valida la arquitectura y la optimización cinemática en 2D. No
es todavía un controlador listo para hardware: el twist se obtiene mediante una
proyección cinemática anisótropa de las velocidades longitudinales, pero no se
modelan todavía las fuerzas de contacto. La estabilidad emplea suelo plano,
altura constante y una aproximación del ZMP. El siguiente nivel deberá
introducir dinámica de actuadores, estimación de estado, incertidumbre del
corredor y un modelo identificado de interacción oruga-terreno.
