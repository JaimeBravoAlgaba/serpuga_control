# SERPUGA Control

Prototipo en Python de control predictivo no lineal para un robot formado por
dos orugas reconfigurables. El controlador sigue una trayectoria con referencias
de velocidad lineal y angular, favorece que las orugas permanezcan paralelas y
adapta la envolvente completa del robot al ancho libre de un corredor.

## Qué incluye

- descripción paramétrica de pivotes, huellas, masas y límites de actuación;
- control predictivo mediante el *twist* del centro de la barra
  `[vx, vy, omega]`;
- cinemática inversa analítica que obtiene `[q1, q2, v1, v2]` en cada periodo;
- velocidades de banda firmadas, permitiendo avance y retroceso por oruga;
- NMPC de disparo múltiple implementado con CasADi/IPOPT;
- coste de paralelismo invariante ante una diferencia de 180 grados;
- límite duro configurable sobre la velocidad de las articulaciones de las orugas;
- una única desigualdad geométrica `ancho_robot <= ancho_libre - 2*margen`;
- corredor sintético intercambiable por el futuro estimador láser;
- deslizamiento, soporte y holgura conservados como diagnósticos no restringidos;
- reserva factible comprobada cuando IPOPT agota su presupuesto online;
- aplicación gráfica con configuración completa y perfiles YAML;
- simulación online: cada paso MPC se resuelve, aplica y dibuja inmediatamente;
- teleoperación manual en vivo mediante `vx`, `vy` y `omega`, con las consignas
  de las orugas calculadas automáticamente;
- carga de parámetros, pausa, reanudación y parada sin precalcular la trayectoria
  ejecutada.

## Estructura

| Módulo | Función |
|---|---|
| `config.py` | Dataclasses numéricas consumidas por el controlador |
| `configuration.py` | Esquema de interfaz, validación y perfiles YAML |
| `robot.py` | Geometría, huellas, CoM y soporte |
| `kinematics.py` | Inversa analítica, slip e integración RK4 |
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

- **Robot**: geometría, masas y límites de posición de las articulaciones;
- **Escenario**: anchuras, posición y transición del hueco;
- **Simulación**: estado inicial, duración y referencias constantes `v` y `ω`;
- **MPC**: horizonte, seguimiento, paralelismo, límites de velocidad, anchura y
  opciones de IPOPT.

La simulación corre en tiempo real desde que se carga un perfil. El panel de
**Teleoperación** permite activar **Modo manual**, que deja de resolver el MPC y
aplica directamente `[vx, vy, omega]` al centro de la barra. La cinemática
inversa calcula en vivo `[q1, q2, v1, v2]`, que se muestran en la telemetría.
En manual, la simulación no se detiene por `duration` ni por `stop_x`.

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

El parámetro `mpc.articulation_rate_limit_rps` fija la velocidad articular
máxima admitida por el NMPC. Los perfiles incluidos usan `1.5 rad/s` como valor
inicial ilustrativo y debe ajustarse a los actuadores reales antes de pasar a
hardware.

Se incluyen tres ejemplos editables:

- `default.yaml`: orugas antiparalelas y hueco de 0,61 m;
- `parallel-gap.yaml`: referencia base paralela y hueco que conserva esa forma;
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

Con los parámetros de demostración, el robot anticipa el estrechamiento, hace
compatible su anchura con el hueco y vuelve a favorecer el paralelismo al
salir. Los resultados exactos dependen del equipo y se imprimen como JSON al
terminar cada ejecución. La primera resolución incluye la construcción interna
del solver y suele tardar más que los ciclos siguientes.

## Alcance actual

Este repositorio valida la arquitectura y la optimización cinemática en 2D. No
es todavía un controlador listo para hardware: el *twist* es la orden
cinemática de alto nivel y se mantiene un servo articular ideal dentro de cada
periodo, aunque el NMPC ya limita la velocidad articular media necesaria para
alcanzar la consigna de la inversa. La inversa es exacta en los pivotes
puntuales. Los límites de velocidad de banda, aceleración/par articular, el
deslizamiento y la estabilidad dinámica no se imponen en esta versión. El
siguiente nivel deberá validar esos efectos antes de llevar el controlador a
hardware.
