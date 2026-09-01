"""Tk desktop application for configuring and running SERPUGA online."""

from __future__ import annotations

import tkinter as tk
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
from tkinter import messagebox, simpledialog, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .configuration import (
    FORM_FIELDS,
    ApplicationConfiguration,
    ConfigurationError,
    ConfigurationStore,
    configuration_from_form_values,
    configuration_to_form_values,
)
from .online_visualization import OnlineSimulationPlot
from .runtime import build_runtime
from .simulation import ClosedLoopSession, SimulationLog, TeleoperationCommand


class ScrollableTab(ttk.Frame):
    """Vertically scrollable form container used by each configuration tab."""

    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.content = ttk.Frame(self.canvas, padding=(10, 8, 12, 16))
        self.window = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window, width=event.width)

    def _bind_wheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event: tk.Event) -> None:
        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            direction = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(direction, "units")


class SimulationApplication:
    """Configuration editor and genuinely online MPC simulation window."""

    poll_interval_ms = 30

    def __init__(
        self,
        store: ConfigurationStore,
        initial_profile: str = "default",
        root: tk.Tk | None = None,
    ) -> None:
        self.store = store
        self.root = tk.Tk() if root is None else root
        self.root.title("SERPUGA · Configuración y simulación MPC online")
        self.root.geometry("1700x960")
        self.root.minsize(1180, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.messages: Queue[tuple[str, Any]] = Queue()
        self.stop_event = Event()
        self.run_gate = Event()
        self.worker: Thread | None = None
        self.pending_configuration: ApplicationConfiguration | None = None
        self.pending_reset_teleoperation = False
        self.current_configuration: ApplicationConfiguration | None = None
        self.last_log: SimulationLog | None = None
        self.form_variables: dict[str, tk.StringVar | tk.BooleanVar] = {}
        self.form_widgets: list[tk.Widget] = []
        self.teleoperation_lock = Lock()
        self.teleoperation_command = TeleoperationCommand()
        self._updating_teleoperation = False

        self._configure_style()
        self._build_layout()
        self._refresh_profiles()
        self._load_profile(initial_profile)
        self.root.after(self.poll_interval_ms, self._poll_messages)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))
        style.configure("Subtitle.TLabel", foreground="#667085")
        style.configure("Run.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("Status.TLabel", padding=(8, 5))

    def _build_layout(self) -> None:
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, padding=(14, 10, 14, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="SERPUGA", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Simulación MPC online y configuración centralizada",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w")

        profile_bar = ttk.Frame(header)
        profile_bar.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(profile_bar, text="Configuración:").grid(row=0, column=0, padx=(0, 6))
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(
            profile_bar,
            textvariable=self.profile_var,
            state="readonly",
            width=22,
        )
        self.profile_combo.grid(row=0, column=1, padx=(0, 6))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_selected)
        self.load_button = ttk.Button(
            profile_bar, text="Cargar", command=self._load_selected_profile
        )
        self.load_button.grid(row=0, column=2, padx=3)
        self.save_button = ttk.Button(
            profile_bar, text="Guardar como…", command=self._save_profile
        )
        self.save_button.grid(row=0, column=3, padx=(3, 12))
        self.run_button = ttk.Button(
            profile_bar,
            text="Cargar parámetros",
            style="Run.TButton",
            command=self.start,
        )
        self.run_button.grid(row=0, column=4, padx=3)
        self.pause_button = ttk.Button(
            profile_bar, text="Pausar", command=self.toggle_pause, state="disabled"
        )
        self.pause_button.grid(row=0, column=5, padx=3)
        self.stop_button = ttk.Button(
            profile_bar, text="Detener", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=6, padx=(3, 0))

        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        simulation_frame = ttk.Frame(panes)
        settings_frame = ttk.Frame(panes, width=460)
        panes.add(simulation_frame, weight=3)
        panes.add(settings_frame, weight=1)

        simulation_frame.rowconfigure(0, weight=1)
        simulation_frame.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(10.5, 8.0), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=simulation_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(simulation_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, toolbar_frame, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.pack(side="left")
        self.plot = OnlineSimulationPlot(self.figure, self.canvas.draw)

        settings_frame.rowconfigure(1, weight=1)
        settings_frame.columnconfigure(0, weight=1)
        ttk.Label(
            settings_frame,
            text="Los parámetros se aplican al pulsar Cargar parámetros.",
            style="Subtitle.TLabel",
            padding=(8, 4),
        ).grid(row=0, column=0, sticky="w")
        self.notebook = ttk.Notebook(settings_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self._build_forms()
        self._build_teleoperation_panel(settings_frame)

        status_frame = ttk.Frame(self.root, padding=(12, 2, 12, 8))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Preparado")
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(
            status_frame, textvariable=self.status_var, style="Status.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            status_frame,
            textvariable=self.metrics_var,
            style="Status.TLabel",
        ).grid(row=0, column=1, sticky="e")

    def _build_forms(self) -> None:
        tabs: dict[str, ScrollableTab] = {}
        group_frames: dict[tuple[str, str], ttk.LabelFrame] = {}
        group_rows: dict[tuple[str, str], int] = {}

        for tab_name in ("Robot", "Escenario", "Simulación", "MPC"):
            tab = ScrollableTab(self.notebook)
            self.notebook.add(tab, text=tab_name)
            tabs[tab_name] = tab

        for spec in FORM_FIELDS:
            tab = tabs[spec.tab]
            group_key = (spec.tab, spec.group)
            if group_key not in group_frames:
                group = ttk.LabelFrame(tab.content, text=spec.group, padding=(8, 6))
                group.grid(
                    row=len([key for key in group_frames if key[0] == spec.tab]),
                    column=0,
                    sticky="ew",
                    pady=(0, 9),
                )
                group.columnconfigure(1, weight=1)
                group_frames[group_key] = group
                group_rows[group_key] = 0
                tab.content.columnconfigure(0, weight=1)

            group = group_frames[group_key]
            row = group_rows[group_key]
            if spec.kind == "bool":
                variable = tk.BooleanVar(value=False)
                widget = ttk.Checkbutton(group, text=spec.label, variable=variable)
                widget.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
            else:
                variable = tk.StringVar()
                ttk.Label(group, text=spec.label).grid(
                    row=row, column=0, sticky="w", padx=(0, 8), pady=2
                )
                widget = ttk.Entry(group, textvariable=variable, width=14)
                widget.grid(row=row, column=1, sticky="ew", pady=2)
                ttk.Label(group, text=spec.unit, foreground="#667085").grid(
                    row=row, column=2, sticky="w", padx=(6, 0), pady=2
                )
            self.form_variables[spec.identifier] = variable
            self.form_widgets.append(widget)
            group_rows[group_key] += 1

    def _build_teleoperation_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Teleoperación", padding=(8, 7))
        panel.grid(row=2, column=0, sticky="ew", padx=2, pady=(8, 0))
        panel.columnconfigure(1, weight=1)

        self.teleop_enabled_var = tk.BooleanVar(value=False)
        toggle = ttk.Checkbutton(
            panel,
            text="Modo manual (desactiva MPC)",
            variable=self.teleop_enabled_var,
            command=self._on_teleoperation_toggled,
        )
        toggle.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self.teleop_twist_vars = [
            tk.DoubleVar(value=0.0),
            tk.DoubleVar(value=0.0),
            tk.DoubleVar(value=0.0),
        ]
        self.teleop_twist_scales: list[ttk.Scale] = []

        rows = (
            ("vx barra", "m/s", self.teleop_twist_vars[0]),
            ("vy barra", "m/s", self.teleop_twist_vars[1]),
            ("omega", "rad/s", self.teleop_twist_vars[2]),
        )
        for row_index, (label, unit, variable) in enumerate(rows, start=1):
            ttk.Label(panel, text=label).grid(
                row=row_index, column=0, sticky="w", padx=(0, 7), pady=2
            )
            scale = ttk.Scale(
                panel,
                variable=variable,
                orient="horizontal",
                command=lambda _value: self._sync_teleoperation_command(),
            )
            scale.grid(row=row_index, column=1, sticky="ew", pady=2)
            self.teleop_twist_scales.append(scale)
            entry = ttk.Entry(panel, textvariable=variable, width=8)
            entry.grid(row=row_index, column=2, sticky="e", padx=(8, 3), pady=2)
            ttk.Label(panel, text=unit, foreground="#667085").grid(
                row=row_index, column=3, sticky="w", pady=2
            )
            variable.trace_add(
                "write",
                lambda *_args: self._sync_teleoperation_command(),
            )

        ttk.Label(
            panel,
            text="La IK calcula q1, q2, v1 y v2 en cada periodo.",
            foreground="#667085",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 1))

        buttons = ttk.Frame(panel)
        buttons.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        buttons.columnconfigure(0, weight=1)
        ttk.Button(
            buttons,
            text="Detener movimiento",
            command=self._stop_manual_motion,
        ).grid(row=0, column=0, sticky="ew")

    def _configure_teleoperation_controls(
        self,
        configuration: ApplicationConfiguration,
        *,
        reset_values: bool,
    ) -> None:
        linear_limit = float(configuration.mpc.body_speed_limit)
        yaw_limit = float(configuration.mpc.body_yaw_rate_limit)

        self._updating_teleoperation = True
        try:
            self.teleop_twist_scales[0].configure(
                from_=-linear_limit,
                to=linear_limit,
            )
            self.teleop_twist_scales[1].configure(
                from_=-linear_limit,
                to=linear_limit,
            )
            self.teleop_twist_scales[2].configure(
                from_=-yaw_limit,
                to=yaw_limit,
            )
            if reset_values:
                for variable in self.teleop_twist_vars:
                    variable.set(0.0)
                self.teleop_enabled_var.set(False)
            else:
                for variable in self.teleop_twist_vars[0:2]:
                    variable.set(
                        float(np.clip(variable.get(), -linear_limit, linear_limit))
                    )
                self.teleop_twist_vars[2].set(
                    float(
                        np.clip(
                            self.teleop_twist_vars[2].get(),
                            -yaw_limit,
                            yaw_limit,
                        )
                    )
                )
        finally:
            self._updating_teleoperation = False
        self._sync_teleoperation_command()

    def _stop_manual_motion(self) -> None:
        for variable in self.teleop_twist_vars:
            variable.set(0.0)

    def _on_teleoperation_toggled(self) -> None:
        if self.teleop_enabled_var.get():
            self.status_var.set("Teleoperación manual · MPC desactivado")
        else:
            self.status_var.set("MPC activado")
        self._sync_teleoperation_command()
        if self.teleop_enabled_var.get() and (
            self.worker is None or not self.worker.is_alive()
        ):
            self.start()

    def _sync_teleoperation_command(self) -> None:
        if self._updating_teleoperation:
            return
        try:
            body_twist = np.array(
                [variable.get() for variable in self.teleop_twist_vars],
                dtype=float,
            )
        except (ValueError, tk.TclError):
            return
        command = TeleoperationCommand(
            enabled=bool(self.teleop_enabled_var.get()),
            body_twist=body_twist,
        )
        with self.teleoperation_lock:
            self.teleoperation_command = command

    def _current_teleoperation_command(self) -> TeleoperationCommand:
        with self.teleoperation_lock:
            command = self.teleoperation_command
            return TeleoperationCommand(
                enabled=command.enabled,
                body_twist=command.body_twist.copy(),
            )

    def _refresh_profiles(self, select: str | None = None) -> None:
        profiles = self.store.list_profiles()
        self.profile_combo.configure(values=profiles)
        if select is not None and select in profiles:
            self.profile_var.set(select)
        elif not self.profile_var.get() and profiles:
            self.profile_var.set(profiles[0])

    def _set_form_configuration(self, configuration: ApplicationConfiguration) -> None:
        values = configuration_to_form_values(configuration)
        for identifier, value in values.items():
            self.form_variables[identifier].set(value)
        self._request_run(configuration, reset_teleoperation=True)

    def _form_configuration(self) -> ApplicationConfiguration:
        values = {
            identifier: variable.get()
            for identifier, variable in self.form_variables.items()
        }
        return configuration_from_form_values(values)

    def _load_profile(self, name_or_path: str) -> None:
        try:
            configuration = self.store.load(name_or_path)
        except ConfigurationError as error:
            profiles = self.store.list_profiles()
            if profiles and name_or_path not in profiles:
                self.profile_var.set(profiles[0])
                configuration = self.store.load(profiles[0])
            else:
                raise RuntimeError(str(error)) from error
        self.profile_var.set(str(name_or_path).rsplit("/", 1)[-1].split(".")[0])
        self._set_form_configuration(configuration)

    def _load_selected_profile(self) -> None:
        name = self.profile_var.get()
        if not name:
            return
        try:
            self._load_profile(name)
        except (ConfigurationError, RuntimeError) as error:
            messagebox.showerror(
                "Configuración no válida", str(error), parent=self.root
            )

    def _on_profile_selected(self, _event: tk.Event) -> None:
        self._load_selected_profile()

    def _save_profile(self) -> None:
        try:
            configuration = self._form_configuration()
        except ConfigurationError as error:
            messagebox.showerror(
                "Configuración no válida", str(error), parent=self.root
            )
            return
        suggested = self.profile_var.get() or "mi-configuracion"
        name = simpledialog.askstring(
            "Guardar configuración",
            "Nombre del perfil:",
            initialvalue=suggested,
            parent=self.root,
        )
        if not name:
            return
        try:
            safe_name = self.store.safe_name(name)
            candidate = self.store.directory / f"{safe_name}.yaml"
            if candidate.exists() and not messagebox.askyesno(
                "Sobrescribir configuración",
                f"Ya existe {candidate.name}. ¿Quieres sobrescribirla?",
                parent=self.root,
            ):
                return
            path = self.store.save(name, configuration)
        except (ConfigurationError, OSError) as error:
            messagebox.showerror("No se pudo guardar", str(error), parent=self.root)
            return
        self._refresh_profiles(path.stem)
        self.status_var.set(f"Configuración guardada en {path}")

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.pause_button.configure(state=state)
        self.stop_button.configure(state=state)

    def start(self) -> None:
        try:
            configuration = self._form_configuration()
        except ConfigurationError as error:
            messagebox.showerror(
                "Configuración no válida", str(error), parent=self.root
            )
            return
        self._request_run(configuration, reset_teleoperation=False)

    def _request_run(
        self,
        configuration: ApplicationConfiguration,
        *,
        reset_teleoperation: bool,
    ) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.pending_configuration = configuration
            self.pending_reset_teleoperation = reset_teleoperation
            self.stop_event.set()
            self.run_gate.set()
            self.pause_button.configure(text="Pausar", state="disabled")
            self.stop_button.configure(state="disabled")
            self.status_var.set("Cargando parámetros tras la iteración actual…")
            return

        self._begin_run(configuration, reset_teleoperation=reset_teleoperation)

    def _begin_run(
        self,
        configuration: ApplicationConfiguration,
        *,
        reset_teleoperation: bool,
    ) -> None:
        self.current_configuration = configuration
        self.last_log = None
        self.plot.reset(configuration)
        self._configure_teleoperation_controls(
            configuration,
            reset_values=reset_teleoperation,
        )
        self.stop_event.clear()
        self.run_gate.set()
        self._set_run_controls_enabled(True)
        self.pause_button.configure(text="Pausar")
        self.status_var.set("Inicializando el problema MPC…")
        self.metrics_var.set("")
        self.worker = Thread(
            target=self._run_online,
            args=(configuration,),
            name="serpuga-online-mpc",
            daemon=True,
        )
        self.worker.start()

    def _run_online(self, configuration: ApplicationConfiguration) -> None:
        try:
            runtime = build_runtime(configuration)
            session = ClosedLoopSession(
                controller=runtime.controller,
                model=runtime.model,
                robot=runtime.robot,
                corridor=configuration.corridor,
                trajectory=runtime.trajectory,
                mpc_parameters=configuration.mpc,
                simulation_parameters=configuration.simulation,
            )
            self.messages.put(("ready", None))
            deadline = monotonic()
            while not self.stop_event.is_set() and not session.finished:
                while not self.run_gate.wait(0.05):
                    if self.stop_event.is_set():
                        break
                    deadline = monotonic()
                if self.stop_event.is_set():
                    break

                started = monotonic()
                command = self._current_teleoperation_command()
                succeeded = session.step(
                    command,
                    stop_when_complete=not command.enabled,
                )
                elapsed = monotonic() - started
                if session.times:
                    frame_drawn = Event()
                    self.messages.put(
                        ("frame", (session.to_log(), elapsed, frame_drawn))
                    )
                    self._wait_for_frame_draw(frame_drawn)
                if not succeeded:
                    break

                deadline += configuration.mpc.dt
                remaining = deadline - monotonic()
                if remaining > 0.0:
                    self.stop_event.wait(remaining)
                else:
                    deadline = monotonic()

            log = session.to_log() if session.times else None
            if session.failure_status is not None:
                self.messages.put(("failure", session.failure_status))
            self.messages.put(("done", log))
        except Exception as error:  # noqa: BLE001 - GUI boundary surfaces errors.
            self.messages.put(("error", str(error)))

    def _wait_for_frame_draw(self, frame_drawn: Event) -> None:
        while not frame_drawn.wait(0.05) and not self.stop_event.is_set():
            continue

    def toggle_pause(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        if self.run_gate.is_set():
            self.run_gate.clear()
            self.pause_button.configure(text="Reanudar")
            self.status_var.set("Simulación pausada")
        else:
            self.run_gate.set()
            self.pause_button.configure(text="Pausar")
            self.status_var.set("Simulación online")

    def stop(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        self.stop_event.set()
        self.run_gate.set()
        self.status_var.set("Deteniendo tras la iteración MPC actual…")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "ready":
                    self.status_var.set("Simulación online · resolviendo paso a paso")
                elif kind == "frame":
                    log, elapsed, frame_drawn = payload
                    self.last_log = log
                    try:
                        self.plot.update_log(log)
                        dt = self.current_configuration.mpc.dt
                        realtime_factor = min(1.0, dt / max(elapsed, dt))
                        mode = log.control_modes[-1] if log.control_modes else "mpc"
                        mode_text = "manual" if mode == "manual" else "mpc"
                        self.metrics_var.set(
                            f"t={log.times[-1] + dt:.2f}s · modo={mode_text} · "
                            f"solve={elapsed:.3f}s · ritmo={realtime_factor:.2f}×"
                        )
                    finally:
                        frame_drawn.set()
                elif kind == "failure":
                    self.status_var.set("El solver MPC no encontró una solución")
                    messagebox.showerror("Fallo del MPC", payload, parent=self.root)
                elif kind == "error":
                    self.status_var.set("No se pudo iniciar la simulación")
                    messagebox.showerror(
                        "Error de simulación", payload, parent=self.root
                    )
                    self._finish_run(None)
                elif kind == "done":
                    self._finish_run(payload)
        except Empty:
            pass
        self.root.after(self.poll_interval_ms, self._poll_messages)

    def _finish_run(self, log: SimulationLog | None) -> None:
        self.last_log = log
        self.worker = None
        self.run_gate.clear()
        if self.pending_configuration is not None:
            configuration = self.pending_configuration
            reset_teleoperation = self.pending_reset_teleoperation
            self.pending_configuration = None
            self.pending_reset_teleoperation = False
            self._begin_run(
                configuration,
                reset_teleoperation=reset_teleoperation,
            )
            return

        self._set_run_controls_enabled(False)
        self.pause_button.configure(text="Pausar")
        if log is None:
            return
        if self.stop_event.is_set():
            self.status_var.set("Simulación detenida por el usuario")
        elif log.completed:
            summary = log.summary()
            self.status_var.set("Simulación finalizada correctamente")
            self.metrics_var.set(
                f"RMSE={summary['position_rmse_m']:.3f}m · "
                f"clear={summary['minimum_clearance_m']:.3f}m · "
                f"diag. slip lat={summary['maximum_lateral_slip_mps']:.3f}m/s"
            )
        else:
            self.status_var.set("Simulación finalizada sin alcanzar el objetivo")

    def _on_close(self) -> None:
        self.stop_event.set()
        self.run_gate.set()
        self.root.destroy()

    def show(self) -> None:
        self.root.mainloop()


def launch_application(
    store: ConfigurationStore,
    initial_profile: str = "default",
) -> None:
    application = SimulationApplication(store=store, initial_profile=initial_profile)
    application.show()
