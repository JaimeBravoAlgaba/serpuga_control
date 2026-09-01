# SERPUGA Control

Prototipo en Python de control predictivo no lineal para un robot formado por dos orugas reconfigurables. El controlador sigue una trayectoria, favorece el paralelismo de las orugas y adapta la envolvente completa del robot al ancho libre del corredor.

## Arquitectura actual

La salida directa del MPC es ahora

```text
[q1_cmd, q2_cmd, v1, v2]
```

Estas cuatro variables son las consignas de actuación. Ya no existe una etapa de cinemática inversa entre el MPC y las orugas.

A partir de esas consignas, `KinematicModel` calcula el movimiento rígido resultante del conjunto mediante cinemática directa. Para cada oruga se plantea la velocidad deseada de su pivote como

```text
v_i [cos(q_i), sin(q_i)]
```

y se obtiene el twist `[vx, vy, omega]` que mejor satisface las cuatro ecuaciones de velocidad mediante mínimos cuadrados regularizados. Ese twist es una magnitud derivada usada para predecir la pose y evaluar el seguimiento de referencia.

El estado del MPC permanece

```text
[X, Y, psi, q1, q2]
```

mientras que el control tiene dimensión 4.

## Restricciones principales

- límites articulares de `q1` y `q2`;
- límite de velocidad articular `|q_cmd - q| / dt`;
- límite de velocidad longitudinal de cada oruga;
- límites sobre velocidad lineal y angular del cuerpo derivadas de la cinemática directa;
- una única restricción geométrica de anchura de la formación frente al hueco.

El coste conserva seguimiento de posición, orientación, velocidad lineal y velocidad angular, además del término de paralelismo

```text
sin²(q1 - q2)
```

que considera equivalentes configuraciones paralelas y antiparalelas.

## Estructura

| Módulo | Función |
|---|---|
| `config.py` | Parámetros numéricos del robot y del MPC |
| `configuration.py` | Perfiles YAML y parámetros de interfaz |
| `robot.py` | Geometría, huellas, CoM y soporte |
| `kinematics.py` | Cinemática directa desde `[q1,q2,v1,v2]` e integración RK4 |
| `trajectory.py` | Referencias y previsualización |
| `corridor.py` | Modelo sintético del espacio libre |
| `nmpc.py` | Formulación y resolución del NMPC |
| `simulation.py` | Simulación online y métricas |
| `app.py` | Aplicación de escritorio |

La formulación matemática completa se resume en [`docs/model.md`](docs/model.md).

## Instalación

```bash
conda env create -f environment.yml
conda activate serpuga-control
```

O alternativamente:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Ejecución

```bash
python -m serpuga_control
python -m serpuga_control --config default --headless
```

Para exportar resultados:

```bash
python -m serpuga_control --config default --screenshot artifacts/gap.png
python -m serpuga_control --config default --video artifacts/run.mp4
```

## Teleoperación

El modo manual conserva temporalmente los tres mandos `[vx, vy, omega]` de la interfaz anterior únicamente por compatibilidad de GUI. En ese modo no se usa el MPC: las articulaciones se mantienen en su posición actual y se asignan velocidades de banda sobre los ejes existentes. Esta compatibilidad no forma parte de la arquitectura del NMPC.

## Pruebas

```bash
pytest
```

## Alcance actual

El modelo sigue siendo cinemático. La pose se integra a partir del twist rígido obtenido de las consignas directas de las orugas y se supone que `q_cmd` se alcanza dentro del periodo de control, limitado por la velocidad articular configurada. No se modelan todavía aceleraciones, pares articulares, dinámica interna de actuadores ni dinámica completa de contacto.
