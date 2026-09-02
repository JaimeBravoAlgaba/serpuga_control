# SERPUGA Control

Prototipo en Python de control predictivo no lineal para un robot formado por dos orugas reconfigurables. El controlador sigue una trayectoria, favorece el paralelismo de las orugas y adapta la envolvente completa del robot al espacio libre del corredor.

## Arquitectura actual

La salida directa del MPC es

```text
[q1_cmd, q2_cmd, v1, v2]
```

Estas cuatro variables son las consignas de actuación. No existe una etapa de cinemática inversa entre el MPC y las orugas.

Para unos ángulos instantáneos `q1`, `q2` y velocidades de banda `v1`, `v2`, `KinematicModel` plantea para cada oruga la velocidad longitudinal deseada de su pivote como

```text
v_i [cos(q_i), sin(q_i)]
```

y obtiene el twist rígido `[vx, vy, omega]` que mejor satisface las cuatro ecuaciones escalares mediante mínimos cuadrados regularizados. Ese twist es una magnitud derivada usada para predecir la pose y evaluar el seguimiento de referencia.

El estado del MPC es

```text
[X, Y, psi, q1, q2]
```

y el control tiene dimensión 4.

Las consignas articulares son objetivos de posición al final del periodo. Dentro de cada intervalo las articulaciones evolucionan linealmente desde el estado medido hasta `q_cmd`; la integración RK4 evalúa la cinemática con los ángulos reales de cada etapa, evitando tratar la reconfiguración como un salto instantáneo.

## Restricciones principales

- límites articulares de `q1` y `q2`;
- límite duro de velocidad articular `|q_cmd - q| / dt`;
- límite de velocidad longitudinal de cada oruga;
- límites sobre velocidad lineal y angular del cuerpo, comprobados al inicio, mitad y final de cada periodo;
- una única desigualdad geométrica de despeje frente a las paredes físicas del corredor, evaluada también a mitad del periodo.

La restricción del corredor se construye a partir de todos los vértices de la huella completa. Para cada vértice se evalúa su distancia lateral al `centre_y` real del corredor y el ancho disponible en su coordenada `x`; los residuos se condensan mediante un máximo suavizado en una única desigualdad escalar.

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
| `kinematics.py` | Cinemática directa desde `[q1_cmd,q2_cmd,v1,v2]` e integración RK4 |
| `trajectory.py` | Referencias y previsualización |
| `corridor.py` | Modelo sintético del espacio libre y residual de despeje |
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

Los perfiles YAML permiten configurar, entre otros parámetros, `articulation_rate_limit_rps` y `track_speed_limit_mps`.

## Teleoperación

El modo manual conserva temporalmente los tres mandos `[vx, vy, omega]` de la interfaz anterior únicamente por compatibilidad de GUI. En ese modo no se usa el MPC: las articulaciones se mantienen en su posición actual y se asignan velocidades de banda sobre los ejes existentes. Esta compatibilidad no forma parte de la arquitectura del NMPC.

## Pruebas

```bash
pytest
```

## Alcance actual

El modelo sigue siendo cinemático. La orientación articular cambia de forma continua dentro de cada periodo, pero no se modelan aceleraciones, pares articulares, dinámica interna de los actuadores ni dinámica completa de contacto. La incompatibilidad entre las velocidades longitudinales solicitadas por ambas orugas y un único movimiento rígido se conserva como diagnóstico de slip mediante el residuo de mínimos cuadrados.
