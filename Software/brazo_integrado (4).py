"""
═══════════════════════════════════════════════════════════════════════════════
  BRAZO ROBÓTICO 3-DOF — SIMULADOR INTEGRADO
  -----------------------------------------------------------------------------
  Pestañas:
    1)  Vista 3D         — Eslabones animados con cinemática directa (DH)
    2)  Motores + PID    — Simulación del lazo de control del firmware .ino
                           (P proporcional con deadband y saturación PWM_MIN/MAX)
    3)  Trayectoria      — Se ingresa una pose final (por ángulos articulares
                           o por coordenadas cartesianas X,Y,Z usando IK
                           analítica) y el brazo se mueve con el perfil
                           quíntico (polinomio grado 5 → velocidad de
                           perfil cúbico: qdd es cúbica en t, jerk finito).
                           Condiciones: q, qd y qdd = 0 en los dos extremos.
  -----------------------------------------------------------------------------
  Parámetros físicos consistentes con los códigos originales:
    DH:      a = [40, 180, 140] mm,  d = [0, 0, 0] mm,  α = [90°, 0°, 0°]
             (a1=40mm: offset horizontal del eje del primer brazo —
              gira con q1, desplazado 40mm del centro de la base)
    PPR:     M1=M2 = 6720 p/rev,  M3 = 8640 p/rev
    PWM:     MIN=60, MAX=180, deadband=20 pulsos
═══════════════════════════════════════════════════════════════════════════════
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registro proyección 3D)

# ═══════════════════════════════════════════════════════════════════════════════
#  TEMA VISUAL  (igual al sim_tray.py)
# ═══════════════════════════════════════════════════════════════════════════════
BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
TEXT     = "#c9d1d9"
ACCENT   = "#f0f6fc"
MUTED    = "#8b949e"
BTN_BG   = "#21262d"
BTN_HOV  = "#388bfd"
BTN_PLAY = "#238636"
BTN_PAUS = "#9e6a03"
BTN_RSET = "#6e4091"
BTN_DIS  = "#1c2128"

C_J = ["#58a6ff", "#3fb950", "#f78166"]          # colores q1, q2, q3
JNAMES = ["q1", "q2", "q3"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   ACCENT,
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.7,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         8,
})

# ═══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DEL ROBOT
# ═══════════════════════════════════════════════════════════════════════════════
# Denavit-Hartenberg (idénticos a sim_tray.py)
_DH_A     = np.array([40.0,  180.0, 140.0])     # a1=40mm: offset horizontal del primer brazo desde el eje de la base
_DH_D     = np.array([0.0,     0.0,   0.0])     # mm
_DH_ALPHA = np.array([np.pi/2, 0.0,   0.0])     # rad

# Límites articulares (grados) — holgados para permitir q0/qf del sim original
Q_MIN = np.array([-180.0, -180.0, -180.0])
Q_MAX = np.array([ 270.0,  180.0,  180.0])

# Parámetros del firmware (brazo_rtos.ino / siservia.ino)
PPR_M12   = 6720.0        # pulsos por vuelta motores 1 y 2
PPR_M3    = 8640.0        # pulsos por vuelta motor 3
PPR       = np.array([PPR_M12, PPR_M12, PPR_M3])
DEADBAND  = 20            # pulsos — zona muerta del controlador P
PWM_MAX   = 180
PWM_MIN   = 60
PWM_RANGE_PULSES = 2000   # map(err, DEADBAND, 2000, PWM_MIN, PWM_MAX)

# Poses por defecto (mismas del sim_tray.py)
Q0_DEF = np.array([  0.0, 117.0, -118.0])
QF_DEF = np.array([245.0,  83.0,  -56.0])

# Discretización para la trayectoria cúbica
T_TOTAL_DEF = 5.0
DT          = 0.01


# ═══════════════════════════════════════════════════════════════════════════════
#  CINEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════
def dh_matrix(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,     sa,     ca,    d],
        [0.,     0.,     0.,   1.],
    ])


def fk(q_deg):
    """Cinemática directa: devuelve lista de orígenes [O0, O1, O2, O3]."""
    q = np.deg2rad(q_deg)
    T = np.eye(4)
    origins = [T[:3, 3].copy()]
    for i in range(3):
        Ti = dh_matrix(_DH_A[i], _DH_D[i], _DH_ALPHA[i], q[i])
        T  = T @ Ti
        origins.append(T[:3, 3].copy())
    return np.array(origins)        # (4, 3)


def ee_pos(q_deg):
    return fk(q_deg)[-1]


# ═══════════════════════════════════════════════════════════════════════════════
#  CINEMÁTICA INVERSA (analítica)
#
#  Dado un punto objetivo (x, y, z) en el marco del mundo, devuelve los tres
#  ángulos articulares (q1, q2, q3) en grados. El robot es 3-DOF con:
#     q1: rotación de la base alrededor de Z₀ (vertical)
#     a₁: offset horizontal desde el eje de la base al eje del hombro
#     q2: hombro,  L₂ = a₂
#     q3: codo,    L₃ = a₃
#
#  Hay 2 soluciones geométricas (codo arriba / codo abajo). Se devuelve la
#  elegida por `elbow_up`. Si el punto está fuera del espacio de trabajo
#  (más lejos que L₂+L₃ o más cerca que |L₂−L₃|), devuelve None.
# ═══════════════════════════════════════════════════════════════════════════════
def ik(x, y, z, elbow_up=True):
    a1 = _DH_A[0]        # offset horizontal
    L2 = _DH_A[1]        # brazo
    L3 = _DH_A[2]        # antebrazo

    # q1: la base apunta hacia el objetivo en el plano X-Y
    q1 = np.arctan2(y, x)

    # En el plano vertical del brazo (tras rotar por q1):
    #   r = radio horizontal desde el eje del hombro
    #   s = altura (z)
    r_total = np.hypot(x, y)
    r = r_total - a1
    s = z

    # Verificar alcance
    L = np.hypot(r, s)
    if L > (L2 + L3) + 1e-9 or L < abs(L2 - L3) - 1e-9:
        return None      # fuera del espacio de trabajo

    # Ley de cosenos: ángulo del codo
    D = (r*r + s*s - L2*L2 - L3*L3) / (2.0 * L2 * L3)
    D = np.clip(D, -1.0, 1.0)
    sign = 1.0 if elbow_up else -1.0
    q3 = sign * np.arctan2(np.sqrt(1.0 - D*D), D)

    # Ángulo del hombro
    q2 = np.arctan2(s, r) - np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))

    return np.rad2deg(np.array([q1, q2, q3]))


def workspace_bounds():
    """Radios mínimo y máximo del espacio de trabajo, y altura máxima."""
    a1, L2, L3 = _DH_A
    r_max = a1 + (L2 + L3)
    r_min = max(0.0, a1 - (L2 + L3))     # si a1 < L2+L3, el robot puede llegar a 0
    # Altura (en el plano del brazo): el alcance vertical desde el hombro es L2+L3
    z_max =  (L2 + L3)
    z_min = -(L2 + L3)
    return r_min, r_max, z_min, z_max


# ═══════════════════════════════════════════════════════════════════════════════
#  PERFIL QUÍNTICO  (polinomio de grado 5 — "perfil de velocidad cúbico")
#
#  Condiciones de frontera:
#    q(0)   = q0        q(tf)   = qf
#    qd(0)  = 0         qd(tf)  = 0
#    qdd(0) = 0         qdd(tf) = 0
#
#  Resultado:
#    q(t)   = polinomio grado 5  (rampa suavísima en S)
#    qd(t)  = polinomio grado 4  (pico simétrico, perfil tipo campana)
#    qdd(t) = polinomio grado 3  (CÚBICA en el tiempo  ← de aquí el nombre)
#    qddd(t)= parabólico         (jerk finito, suave)
#
#  Coeficientes (con D = qf - q0):
#    a0 = q0,  a1 = a2 = 0
#    a3 =  10·D/tf³
#    a4 = -15·D/tf⁴
#    a5 =   6·D/tf⁵
# ═══════════════════════════════════════════════════════════════════════════════
def perfil_quintico(q0, qf, t, tf):
    n = len(t)
    q, qd, qdd = np.zeros((3, n)), np.zeros((3, n)), np.zeros((3, n))
    for j in range(3):
        D = qf[j] - q0[j]
        a3 =  10.0 * D / tf**3
        a4 = -15.0 * D / tf**4
        a5 =   6.0 * D / tf**5
        q[j]   = q0[j] + a3*t**3 +   a4*t**4 +   a5*t**5
        qd[j]  =        3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        qdd[j] =        6*a3*t   + 12*a4*t**2 + 20*a5*t**3
    return q, qd, qdd


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN DEL LAZO DE CONTROL DEL FIRMWARE
# ═══════════════════════════════════════════════════════════════════════════════
def sim_firmware_loop(q_start_deg, q_ref_deg, dt=0.02, t_max=6.0,
                      k_planta=0.6):
    """
    Replica el lazo del .ino (controlMotor / tareaMotor):
        error_pulsos = posRef - pos
        si |error| <= DEADBAND  -> motor para
        sino  PWM = map(|error|, DEADBAND, 2000, PWM_MIN, PWM_MAX)  (saturado)
        sentido = sign(error)

    La planta se modela como integrador de primer orden:
        v_pulsos = k_planta * PWM * sign(error)     [pulsos/ciclo]
        pos(k+1) = pos(k) + v_pulsos * dt_ratio

    Retorna arreglos de tiempo, posiciones (deg), errores (deg) y PWM firmados.
    El parámetro k_planta ajusta qué tan "rápido" responde el motor simulado.
    """
    n = int(t_max / dt) + 1
    t = np.arange(n) * dt

    # A pulsos
    pos_ref_p = (q_ref_deg   / 360.0) * PPR
    pos_p     = (q_start_deg / 360.0) * PPR.copy()

    pos_hist   = np.zeros((3, n))
    pwm_hist   = np.zeros((3, n))
    err_hist   = np.zeros((3, n))   # en pulsos (para graficar se convierte a deg)

    for k in range(n):
        for j in range(3):
            error = pos_ref_p[j] - pos_p[j]
            abserr = abs(error)
            if abserr <= DEADBAND:
                pwm = 0
            else:
                # map(abserr, DEADBAND, 2000, PWM_MIN, PWM_MAX)
                x = (abserr - DEADBAND) / (PWM_RANGE_PULSES - DEADBAND)
                pwm = PWM_MIN + x * (PWM_MAX - PWM_MIN)
                pwm = max(PWM_MIN, min(PWM_MAX, pwm))

            signo = 1.0 if error > 0 else (-1.0 if error < 0 else 0.0)

            # Planta simplificada
            v_pulsos_por_s = k_planta * pwm * signo   # pulsos/s
            pos_p[j] += v_pulsos_por_s * dt

            pos_hist[j, k] = pos_p[j]
            pwm_hist[j, k] = pwm * signo
            err_hist[j, k] = error

    # Convertir a grados
    pos_deg_hist = pos_hist / PPR[:, None] * 360.0
    err_deg_hist = err_hist / PPR[:, None] * 360.0
    return t, pos_deg_hist, pwm_hist, err_deg_hist


# ═══════════════════════════════════════════════════════════════════════════════
#  WIDGETS AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════
def make_labeled_entry(parent, label, default, width=7):
    frm = tk.Frame(parent, bg=PANEL)
    tk.Label(frm, text=label, bg=PANEL, fg=MUTED,
             font=("Courier New", 9)).pack(side="left", padx=(0, 4))
    var = tk.StringVar(value=str(default))
    e = tk.Entry(frm, textvariable=var, width=width,
                 bg="#0a0e14", fg=ACCENT, insertbackground=ACCENT,
                 relief="flat", font=("Courier New", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=BTN_HOV)
    e.pack(side="left")
    return frm, var


# ═══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class BrazoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Brazo Robótico 3-DOF — Simulador Integrado")
        self.configure(bg=BG)
        try:
            self.state("zoomed")
        except tk.TclError:
            self.geometry("1400x850")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._tab = 0
        self._job = None

        # Estado articular actual (se usa en la pestaña 3D y como q_inicio en trayectoria)
        self._q_current = Q0_DEF.copy()

        self._build_header()
        self._build_tabs()
        self._build_statusbar()
        self._build_figures()

        self._switch_tab(0)

    # ──────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=BG, pady=6)
        hdr.pack(fill="x", side="top")
        tk.Label(hdr, text="BRAZO ROBOTICO 3-DOF  —  SIMULADOR INTEGRADO",
                 bg=BG, fg=ACCENT, font=("Courier New", 13, "bold")).pack()
        tk.Label(hdr,
                 text="DH: a=[40, 180, 140] mm  d=[0, 0, 0] mm  alpha=[90,0,0] deg   |   "
                      "PPR: M1/M2=6720  M3=8640   |   PWM 60-180  deadband=20",
                 bg=BG, fg=MUTED, font=("Courier New", 8)).pack()
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="top")

    def _build_tabs(self):
        bar = tk.Frame(self, bg=PANEL, pady=4)
        bar.pack(fill="x", side="top")
        self._tab_btns = []
        labels = ["  1. Vista 3D  ", "  2. Motores + PID  ", "  3. Trayectoria quintica (vel cubica)  "]
        for i, lbl in enumerate(labels):
            b = tk.Button(bar, text=lbl,
                          bg=BTN_HOV if i == 0 else BTN_BG,
                          fg=ACCENT if i == 0 else TEXT,
                          relief="flat", bd=0,
                          font=("Courier New", 10, "bold"),
                          cursor="hand2", padx=14, pady=7,
                          command=lambda i=i: self._switch_tab(i))
            b.pack(side="left", padx=4)
            self._tab_btns.append(b)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="top")

    def _build_statusbar(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")
        sb = tk.Frame(self, bg="#0a0e14", pady=3)
        sb.pack(fill="x", side="bottom")
        self._lbl_status = tk.Label(
            sb, text="Listo.", bg="#0a0e14", fg=MUTED,
            font=("Courier New", 8), anchor="w")
        self._lbl_status.pack(side="left", padx=10)

    # ──────────────────────────────────────────────────────────
    #  FIGURAS
    # ──────────────────────────────────────────────────────────
    def _build_figures(self):
        self._main = tk.Frame(self, bg=BG)
        self._main.pack(fill="both", expand=True, side="top")

        # ---- Frames por pestaña ------------------------------------------------
        self._frame_3d   = tk.Frame(self._main, bg=BG)
        self._frame_pid  = tk.Frame(self._main, bg=BG)
        self._frame_traj = tk.Frame(self._main, bg=BG)

        self._build_tab_3d(self._frame_3d)
        self._build_tab_pid(self._frame_pid)
        self._build_tab_traj(self._frame_traj)

    def _switch_tab(self, idx):
        # Parar cualquier animación pendiente de otras pestañas
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

        for i, b in enumerate(self._tab_btns):
            if i == idx:
                b.config(bg=BTN_HOV, fg=ACCENT)
            else:
                b.config(bg=BTN_BG, fg=TEXT)

        for f in (self._frame_3d, self._frame_pid, self._frame_traj):
            f.pack_forget()
        target = [self._frame_3d, self._frame_pid, self._frame_traj][idx]
        target.pack(fill="both", expand=True)
        self._tab = idx

    # ═══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA 1 — VISTA 3D
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_tab_3d(self, parent):
        # Panel de controles (sliders)
        ctl = tk.Frame(parent, bg=PANEL, pady=10)
        ctl.pack(fill="x", side="bottom")

        tk.Label(ctl, text="Angulos articulares (deg)",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(side="left", padx=(14, 20))

        self._sliders_3d = []
        self._lbl_slider = []
        for j in range(3):
            col = tk.Frame(ctl, bg=PANEL)
            col.pack(side="left", padx=10)
            tk.Label(col, text=JNAMES[j], bg=PANEL,
                     fg=C_J[j], font=("Courier New", 10, "bold")).pack()
            var = tk.DoubleVar(value=self._q_current[j])
            s = tk.Scale(col, from_=Q_MIN[j], to=Q_MAX[j],
                         resolution=1.0, orient="horizontal",
                         length=220, bg=PANEL, fg=TEXT,
                         troughcolor=BTN_BG, highlightthickness=0,
                         font=("Courier New", 8),
                         variable=var,
                         command=lambda _v, j=j: self._on_slider_3d(j))
            s.pack()
            self._sliders_3d.append(var)

        # Botón reset
        tk.Frame(ctl, bg=BORDER, width=2, height=40).pack(side="left", padx=14)
        tk.Button(ctl, text="  Pose inicial  ",
                  bg=BTN_RSET, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=14, pady=6,
                  command=self._reset_pose_3d).pack(side="left", padx=6)

        tk.Button(ctl, text="  Cero (q=0)  ",
                  bg=BTN_BG, fg=TEXT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=14, pady=6,
                  command=self._zero_pose_3d).pack(side="left", padx=6)

        # Info efector final
        self._lbl_ee = tk.Label(ctl, text="", bg=PANEL, fg=ACCENT,
                                font=("Courier New", 10, "bold"))
        self._lbl_ee.pack(side="left", padx=20)

        # Figura 3D
        self._fig3d = plt.Figure(figsize=(12, 7), facecolor=BG)
        self._ax3d  = self._fig3d.add_subplot(111, projection="3d")
        self._fig3d.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

        self._canvas3d = FigureCanvasTkAgg(self._fig3d, master=parent)
        self._canvas3d.get_tk_widget().pack(fill="both", expand=True, side="top")

        self._draw_3d(self._q_current)

    def _on_slider_3d(self, j):
        self._q_current[j] = self._sliders_3d[j].get()
        self._draw_3d(self._q_current)

    def _reset_pose_3d(self):
        self._q_current = Q0_DEF.copy()
        for j in range(3):
            self._sliders_3d[j].set(self._q_current[j])
        self._draw_3d(self._q_current)

    def _zero_pose_3d(self):
        self._q_current = np.zeros(3)
        for j in range(3):
            self._sliders_3d[j].set(0.0)
        self._draw_3d(self._q_current)

    def _draw_3d(self, q_deg):
        ax = self._ax3d
        ax.clear()
        ax.set_facecolor("#0a0f16")

        origins = fk(q_deg)                  # (4,3)
        xs, ys, zs = origins[:, 0], origins[:, 1], origins[:, 2]

        # Suelo (malla tenue)
        R = 350
        ax.plot([-R, R], [0, 0], [0, 0], color="#30363d", lw=0.6)
        ax.plot([0, 0], [-R, R], [0, 0], color="#30363d", lw=0.6)

        # Eslabones
        link_colors = ["#8b949e", "#58a6ff", "#3fb950", "#f78166"]
        ax.plot(xs, ys, zs, color="#8b949e", lw=1.0, alpha=0.4)     # traza guía
        for i in range(3):
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [zs[i], zs[i+1]],
                    color=C_J[i], lw=5, solid_capstyle="round")

        # Articulaciones
        ax.scatter(xs[:-1], ys[:-1], zs[:-1], color=ACCENT,
                   s=70, edgecolors="#000", linewidth=1, zorder=5)
        # Efector final
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="#f78166",
                   s=140, marker="*", edgecolors="#000", linewidth=1, zorder=6)

        # Base
        theta = np.linspace(0, 2*np.pi, 40)
        ax.plot(40*np.cos(theta), 40*np.sin(theta), np.zeros_like(theta),
                color=MUTED, lw=1.2)

        # Ejes del mundo (pequeños, en origen)
        L = 60
        ax.plot([0, L], [0, 0], [0, 0], color="#f85149", lw=1.2)
        ax.plot([0, 0], [0, L], [0, 0], color="#3fb950", lw=1.2)
        ax.plot([0, 0], [0, 0], [0, L], color="#58a6ff", lw=1.2)

        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_zlabel("Z (mm)", fontsize=8)
        ax.set_title(f"Pose actual: q1={q_deg[0]:+.1f}°  "
                     f"q2={q_deg[1]:+.1f}°  q3={q_deg[2]:+.1f}°",
                     fontsize=9, color=ACCENT, pad=8)
        ax.tick_params(labelsize=7)
        # Fondo del panel 3D
        ax.xaxis.pane.set_facecolor("#0a0f16")
        ax.yaxis.pane.set_facecolor("#0a0f16")
        ax.zaxis.pane.set_facecolor("#0a0f16")
        ax.xaxis.pane.set_edgecolor(BORDER)
        ax.yaxis.pane.set_edgecolor(BORDER)
        ax.zaxis.pane.set_edgecolor(BORDER)

        ee = origins[-1]
        self._lbl_ee.config(
            text=f"Efector final: ({ee[0]:+7.1f}, {ee[1]:+7.1f}, {ee[2]:+7.1f}) mm")
        self._canvas3d.draw_idle()

    # ═══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA 2 — MOTORES + PID (simulación del lazo del firmware)
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_tab_pid(self, parent):
        # Panel superior de parámetros
        ctl = tk.Frame(parent, bg=PANEL, pady=9)
        ctl.pack(fill="x", side="bottom")

        tk.Label(ctl, text="Posicion inicial (deg):",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(14, 6))
        self._pid_q0 = []
        for j in range(3):
            frm, var = make_labeled_entry(ctl, JNAMES[j], Q0_DEF[j])
            frm.pack(side="left", padx=4); self._pid_q0.append(var)

        tk.Frame(ctl, bg=BORDER, width=2, height=30).pack(side="left", padx=10)

        tk.Label(ctl, text="Referencia (deg):",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(6, 6))
        self._pid_qf = []
        for j in range(3):
            frm, var = make_labeled_entry(ctl, JNAMES[j], QF_DEF[j])
            frm.pack(side="left", padx=4); self._pid_qf.append(var)

        tk.Frame(ctl, bg=BORDER, width=2, height=30).pack(side="left", padx=10)

        frm, self._pid_k = make_labeled_entry(ctl, "k_planta", 6.0, width=6)
        frm.pack(side="left", padx=4)
        tk.Label(ctl, text="(pulsos/s por unidad de PWM)",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(side="left", padx=(0, 8))

        tk.Button(ctl, text="  Simular  ",
                  bg=BTN_PLAY, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=18, pady=7,
                  command=self._run_pid_sim).pack(side="left", padx=12)

        # Figura
        self._fig_pid = plt.Figure(figsize=(16, 8), facecolor=BG)
        self._fig_pid.subplots_adjust(left=0.055, right=0.985, top=0.94,
                                      bottom=0.07, hspace=0.55, wspace=0.33)
        gs = gridspec.GridSpec(3, 3, figure=self._fig_pid)

        self._ax_pid_pos = [self._fig_pid.add_subplot(gs[0, j]) for j in range(3)]
        self._ax_pid_err = [self._fig_pid.add_subplot(gs[1, j]) for j in range(3)]
        self._ax_pid_pwm = [self._fig_pid.add_subplot(gs[2, j]) for j in range(3)]

        for j in range(3):
            self._ax_pid_pos[j].set_title(
                f"Posicion {JNAMES[j]}  (ref vs actual)", fontsize=8, pad=3)
            self._ax_pid_pos[j].set_ylabel("grados", fontsize=7)
            self._ax_pid_err[j].set_title(
                f"Error {JNAMES[j]}", fontsize=8, pad=3)
            self._ax_pid_err[j].set_ylabel("grados", fontsize=7)
            self._ax_pid_pwm[j].set_title(
                f"PWM firmado {JNAMES[j]}  (+/-{PWM_MAX})", fontsize=8, pad=3)
            self._ax_pid_pwm[j].set_ylabel("PWM", fontsize=7)
            self._ax_pid_pwm[j].set_xlabel("t (s)", fontsize=7)

        self._canvas_pid = FigureCanvasTkAgg(self._fig_pid, master=parent)
        self._canvas_pid.get_tk_widget().pack(fill="both", expand=True, side="top")

        # Correr una vez con valores por defecto
        self._run_pid_sim()

    def _run_pid_sim(self):
        try:
            q0 = np.array([float(v.get()) for v in self._pid_q0])
            qf = np.array([float(v.get()) for v in self._pid_qf])
            k  = float(self._pid_k.get())
        except ValueError:
            messagebox.showerror("Entrada invalida",
                                 "Revisa los angulos y k_planta — deben ser numeros.")
            return

        t, pos, pwm, err = sim_firmware_loop(q0, qf, dt=0.02, t_max=15.0,
                                             k_planta=k)

        for j in range(3):
            ax_p = self._ax_pid_pos[j]; ax_p.clear()
            ax_p.axhline(qf[j], color=MUTED, lw=0.8, ls="--", label="ref")
            ax_p.plot(t, pos[j], color=C_J[j], lw=1.6, label="pos")
            # Banda de deadband (en deg)
            db_deg = DEADBAND / PPR[j] * 360.0
            ax_p.fill_between(t, qf[j]-db_deg, qf[j]+db_deg,
                              color=BTN_HOV, alpha=0.10, label="deadband")
            ax_p.set_title(f"Posicion {JNAMES[j]}  (ref={qf[j]:.1f}°)",
                           fontsize=8, pad=3)
            ax_p.set_ylabel("grados", fontsize=7)
            ax_p.legend(fontsize=6, facecolor=PANEL, edgecolor=BORDER,
                        loc="best")

            ax_e = self._ax_pid_err[j]; ax_e.clear()
            ax_e.axhline(0, color=MUTED, lw=0.6, ls=":")
            ax_e.plot(t, err[j], color=C_J[j], lw=1.4)
            ax_e.set_title(f"Error {JNAMES[j]}", fontsize=8, pad=3)
            ax_e.set_ylabel("grados", fontsize=7)

            ax_u = self._ax_pid_pwm[j]; ax_u.clear()
            ax_u.axhline( PWM_MAX, color="#f85149", lw=0.6, ls="--")
            ax_u.axhline(-PWM_MAX, color="#f85149", lw=0.6, ls="--")
            ax_u.axhline( PWM_MIN, color=MUTED, lw=0.5, ls=":")
            ax_u.axhline(-PWM_MIN, color=MUTED, lw=0.5, ls=":")
            ax_u.axhline(0, color=MUTED, lw=0.6, ls=":")
            ax_u.plot(t, pwm[j], color=C_J[j], lw=1.4)
            ax_u.set_ylim(-PWM_MAX*1.15, PWM_MAX*1.15)
            ax_u.set_title(f"PWM firmado {JNAMES[j]}", fontsize=8, pad=3)
            ax_u.set_ylabel("PWM", fontsize=7)
            ax_u.set_xlabel("t (s)", fontsize=7)

        # Resumen tiempo de establecimiento (error dentro de deadband + mantiene)
        tiempos = []
        for j in range(3):
            db_deg = DEADBAND / PPR[j] * 360.0
            dentro = np.abs(err[j]) <= db_deg
            t_set = None
            # último instante en que salió de la banda + dt
            idx = np.where(~dentro)[0]
            if len(idx) == 0:
                t_set = 0.0
            elif idx[-1] < len(t) - 1:
                t_set = t[idx[-1] + 1]
            tiempos.append(t_set)

        self._lbl_status.config(
            text=f"PID sim OK.  "
                 f"t_est q1={tiempos[0] if tiempos[0] is not None else '—'} s   "
                 f"q2={tiempos[1] if tiempos[1] is not None else '—'} s   "
                 f"q3={tiempos[2] if tiempos[2] is not None else '—'} s   "
                 f"(deadband: {DEADBAND} pulsos)")
        self._canvas_pid.draw_idle()

    # ═══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA 3 — TRAYECTORIA QUÍNTICA (perfil de velocidad cúbico)
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_tab_traj(self, parent):
        # Panel de controles (2 filas)
        ctl = tk.Frame(parent, bg=PANEL, pady=6)
        ctl.pack(fill="x", side="bottom")

        # ── Fila 1: selector de modo + entradas ──
        row1 = tk.Frame(ctl, bg=PANEL); row1.pack(fill="x", pady=(2, 4))

        tk.Label(row1, text="Meta:",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(side="left", padx=(14, 6))
        self._tr_mode = tk.StringVar(value="articular")
        for lbl, val in [("Articular (q1,q2,q3)", "articular"),
                         ("Cartesiano (X,Y,Z)", "cartesiano")]:
            tk.Radiobutton(row1, text=lbl, variable=self._tr_mode, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2",
                           command=self._update_traj_entry_visibility
                           ).pack(side="left", padx=3)

        tk.Frame(row1, bg=BORDER, width=2, height=26).pack(side="left", padx=10)

        # Entradas articulares (meta)
        self._frm_art = tk.Frame(row1, bg=PANEL)
        self._frm_art.pack(side="left")
        tk.Label(self._frm_art, text="qf (deg):",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(0, 4))
        self._tr_qf = []
        for j in range(3):
            frm, var = make_labeled_entry(self._frm_art, JNAMES[j], QF_DEF[j])
            frm.pack(side="left", padx=3); self._tr_qf.append(var)

        # Entradas cartesianas (meta)  — inicialmente ocultas
        self._frm_cart = tk.Frame(row1, bg=PANEL)
        ee_def = ee_pos(QF_DEF)   # posición EE correspondiente a QF_DEF por defecto
        tk.Label(self._frm_cart, text="meta (mm):",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(0, 4))
        self._tr_xyz = []
        for axname, default in zip(["X", "Y", "Z"], ee_def):
            frm, var = make_labeled_entry(self._frm_cart, axname, f"{default:.1f}")
            frm.pack(side="left", padx=3); self._tr_xyz.append(var)
        tk.Label(self._frm_cart, text="codo:",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(8, 2))
        self._tr_elbow = tk.StringVar(value="up")
        for lbl, val in [("arriba", "up"), ("abajo", "down")]:
            tk.Radiobutton(self._frm_cart, text=lbl, variable=self._tr_elbow,
                           value=val, bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2"
                           ).pack(side="left", padx=2)

        # ── Fila 2: q inicial + metodo + tf + botones ──
        row2 = tk.Frame(ctl, bg=PANEL); row2.pack(fill="x", pady=(2, 2))

        tk.Label(row2, text="q inicial (deg):",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(14, 6))
        self._tr_q0 = []
        for j in range(3):
            frm, var = make_labeled_entry(row2, JNAMES[j], Q0_DEF[j])
            frm.pack(side="left", padx=3); self._tr_q0.append(var)

        tk.Frame(row2, bg=BORDER, width=2, height=26).pack(side="left", padx=10)

        tk.Label(row2, text="Interp:",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 9, "bold")).pack(side="left", padx=(2, 4))
        self._tr_interp = tk.StringVar(value="articular")
        for lbl, val in [("Articular (curva)", "articular"),
                         ("Cartesiano (recta)", "cartesiano")]:
            tk.Radiobutton(row2, text=lbl, variable=self._tr_interp, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2",
                           ).pack(side="left", padx=2)

        tk.Frame(row2, bg=BORDER, width=2, height=26).pack(side="left", padx=10)

        frm, self._tr_tf = make_labeled_entry(row2, "tf(s)", T_TOTAL_DEF, width=5)
        frm.pack(side="left", padx=4)

        tk.Button(row2, text="  Planificar  ",
                  bg=BTN_HOV, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=14, pady=7,
                  command=self._plan_traj).pack(side="left", padx=8)

        self._btn_play_tr = tk.Button(
            row2, text="  ▶ Animar  ",
            bg=BTN_PLAY, fg=ACCENT,
            relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=14, pady=7,
            command=self._play_traj)
        self._btn_play_tr.pack(side="left", padx=4)

        self._btn_pause_tr = tk.Button(
            row2, text="  ⏸ Pausar  ",
            bg=BTN_DIS, fg=MUTED, state="disabled",
            relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=14, pady=7,
            command=self._pause_traj)
        self._btn_pause_tr.pack(side="left", padx=4)

        tk.Button(row2, text="  ↺ Reset  ",
                  bg=BTN_RSET, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=14, pady=7,
                  command=self._reset_traj).pack(side="left", padx=4)

        # Figura: izquierda 3D, derecha q/qd/qdd + EE
        self._fig_tr = plt.Figure(figsize=(16, 8), facecolor=BG)
        self._fig_tr.subplots_adjust(left=0.03, right=0.985, top=0.95,
                                     bottom=0.07, hspace=0.55, wspace=0.35)
        gs = gridspec.GridSpec(3, 4, figure=self._fig_tr)

        self._ax_tr3d = self._fig_tr.add_subplot(gs[:, 0:2], projection="3d")
        self._ax_tr_q   = self._fig_tr.add_subplot(gs[0, 2])
        self._ax_tr_qd  = self._fig_tr.add_subplot(gs[1, 2])
        self._ax_tr_qdd = self._fig_tr.add_subplot(gs[2, 2])
        self._ax_tr_ee  = self._fig_tr.add_subplot(gs[0, 3])
        self._ax_tr_vee = self._fig_tr.add_subplot(gs[1, 3])
        self._ax_tr_pct = self._fig_tr.add_subplot(gs[2, 3])

        self._canvas_tr = FigureCanvasTkAgg(self._fig_tr, master=parent)
        self._canvas_tr.get_tk_widget().pack(fill="both", expand=True, side="top")

        self._tr_data = None
        self._tr_frame = 0
        self._tr_running = False

        self._update_traj_entry_visibility()
        self._plan_traj()

    def _update_traj_entry_visibility(self):
        """Muestra/oculta las entradas de articular o cartesiano segun el modo."""
        if self._tr_mode.get() == "articular":
            self._frm_cart.pack_forget()
            self._frm_art.pack(side="left")
        else:
            self._frm_art.pack_forget()
            self._frm_cart.pack(side="left")

    # ──────────────────────────────────────────
    def _plan_traj(self):
        try:
            q0 = np.array([float(v.get()) for v in self._tr_q0])
            tf = float(self._tr_tf.get())
            if tf <= 0:
                raise ValueError("tf debe ser > 0")

            if self._tr_mode.get() == "articular":
                qf = np.array([float(v.get()) for v in self._tr_qf])
                src_info = f"qf articular = [{qf[0]:.1f}, {qf[1]:.1f}, {qf[2]:.1f}] deg"
            else:
                # Modo cartesiano: leer X, Y, Z y resolver IK
                xyz = np.array([float(v.get()) for v in self._tr_xyz])
                elbow_up = (self._tr_elbow.get() == "up")
                qf = ik(xyz[0], xyz[1], xyz[2], elbow_up=elbow_up)
                if qf is None:
                    r_min, r_max, _, _ = workspace_bounds()
                    messagebox.showerror(
                        "Fuera de alcance",
                        f"El punto ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm "
                        f"esta fuera del espacio de trabajo.\n\n"
                        f"Radio horizontal (sqrt(X² + Y²)): debe estar entre "
                        f"{max(0.0, _DH_A[0] - (_DH_A[1] + _DH_A[2])):.1f} y "
                        f"{_DH_A[0] + _DH_A[1] + _DH_A[2]:.1f} mm.\n"
                        f"Alcance desde el hombro: < {(_DH_A[1] + _DH_A[2]):.1f} mm.")
                    return
                # Reflejar qf calculado en las casillas articulares (informativo)
                for j in range(3):
                    self._tr_qf[j].set(f"{qf[j]:.2f}")
                src_info = (f"meta cartesiana = ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm "
                            f"-> IK({'up' if elbow_up else 'down'}) = "
                            f"[{qf[0]:.1f}, {qf[1]:.1f}, {qf[2]:.1f}] deg")
        except ValueError as e:
            messagebox.showerror("Entrada invalida", str(e))
            return

        # Validar limites articulares (advertencia, no bloquea)
        for j in range(3):
            if not (Q_MIN[j] <= qf[j] <= Q_MAX[j]) or \
               not (Q_MIN[j] <= q0[j] <= Q_MAX[j]):
                messagebox.showwarning(
                    "Fuera de rango articular",
                    f"{JNAMES[j]} fuera de [{Q_MIN[j]}, {Q_MAX[j]}]  —  "
                    "se continuará pero revisa los valores.")

        # Puntos cartesianos de inicio y fin (via FK de q0 y qf)
        p0 = ee_pos(q0)
        pf = ee_pos(qf)

        t = np.arange(0, tf + DT, DT)
        N = len(t)
        metodo = self._tr_interp.get()

        if metodo == "articular":
            # --- Interpolar los angulos articulares con perfil quintico ---
            q, qd, qdd = perfil_quintico(q0, qf, t, tf)
            pos_ee = np.array([ee_pos(q[:, i]) for i in range(N)])

        else:
            # --- Interpolar X,Y,Z con perfil quintico y resolver IK ---
            # Antes hay que elegir la rama de codo coherente con la pose inicial.
            # La detectamos con el q3 de la pose actual: q3>=0 -> codo "up"
            elbow_up_sel = (q0[2] >= 0.0)

            pos_ee = np.zeros((N, 3))
            for k in range(3):
                # Perfil quintico sobre la coordenada k
                D = pf[k] - p0[k]
                pos_ee[:, k] = (
                    p0[k]
                    + ( 10.0 * D / tf**3) * t**3
                    + (-15.0 * D / tf**4) * t**4
                    + (  6.0 * D / tf**5) * t**5
                )

            # IK punto por punto para obtener q(t)
            q = np.zeros((3, N))
            fuera = 0
            for i in range(N):
                sol = ik(pos_ee[i, 0], pos_ee[i, 1], pos_ee[i, 2],
                         elbow_up=elbow_up_sel)
                if sol is None:
                    fuera += 1
                    # repetir ultimo q valido para no romper la animacion
                    q[:, i] = q[:, i-1] if i > 0 else q0
                else:
                    q[:, i] = sol

            if fuera > 0:
                messagebox.showwarning(
                    "Ruta fuera de alcance",
                    f"{fuera} de {N} puntos de la linea recta caen fuera del "
                    "espacio de trabajo. El efector no podra seguirla exactamente.")

            # Derivar qd y qdd de q(t) por diferencias (no salen polinomios limpios)
            qd  = np.gradient(q,  t, axis=1)
            qdd = np.gradient(qd, t, axis=1)

        v_ee = np.gradient(pos_ee, t, axis=0)
        speed_ee = np.linalg.norm(v_ee, axis=1)

        self._tr_data = dict(
            t=t, q=q, qd=qd, qdd=qdd,
            q0=q0, qf=qf, tf=tf,
            p0=p0, pf=pf,
            pos_ee=pos_ee, v_ee=v_ee, speed_ee=speed_ee,
            metodo=metodo,
        )
        self._tr_frame = 0
        self._draw_traj_static()
        self._update_traj(0)
        self._set_play_buttons(playing=False, finished=False)
        self._lbl_status.config(
            text=f"Planificado ({metodo}).  {src_info}  |  tf={tf:.2f}s  N={N}  |  "
                 f"EE: ({pos_ee[0,0]:.1f},{pos_ee[0,1]:.1f},{pos_ee[0,2]:.1f}) -> "
                 f"({pos_ee[-1,0]:.1f},{pos_ee[-1,1]:.1f},{pos_ee[-1,2]:.1f}) mm")

    def _draw_traj_static(self):
        d = self._tr_data
        if d is None:
            return
        t = d["t"]; q = d["q"]; qd = d["qd"]; qdd = d["qdd"]

        # --- q, qd, qdd ---
        for ax, data, ttl, unit in [
            (self._ax_tr_q,   q,   "Posicion articular",    "deg"),
            (self._ax_tr_qd,  qd,  "Velocidad articular",   "deg/s"),
            (self._ax_tr_qdd, qdd, "Aceleracion articular", "deg/s2"),
        ]:
            ax.clear()
            for j in range(3):
                ax.plot(t, data[j], color=C_J[j], lw=1.5, label=JNAMES[j])
            ax.set_title(ttl, fontsize=8, pad=3)
            ax.set_ylabel(unit, fontsize=7)
            ax.axhline(0, color=MUTED, lw=0.5, ls=":")
        self._ax_tr_qdd.set_xlabel("t (s)", fontsize=7)
        self._ax_tr_q.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER,
                             loc="best", ncol=3)

        # --- trayectoria XYZ del efector ---
        ax = self._ax_tr_ee; ax.clear()
        ax.plot(t, d["pos_ee"][:, 0], color="#f85149", lw=1.4, label="X")
        ax.plot(t, d["pos_ee"][:, 1], color="#3fb950", lw=1.4, label="Y")
        ax.plot(t, d["pos_ee"][:, 2], color="#58a6ff", lw=1.4, label="Z")
        ax.set_title("Posicion efector final", fontsize=8, pad=3)
        ax.set_ylabel("mm", fontsize=7)
        ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor=BORDER, loc="best", ncol=3)

        ax = self._ax_tr_vee; ax.clear()
        ax.plot(t, d["speed_ee"], color="#d2a8ff", lw=1.5)
        ax.set_title("|v| efector final", fontsize=8, pad=3)
        ax.set_ylabel("mm/s", fontsize=7)

        ax = self._ax_tr_pct; ax.clear()
        ax.set_title("Progreso", fontsize=8, pad=3)
        ax.set_xlim(0, d["tf"]); ax.set_ylim(0, 100)
        ax.set_ylabel("%", fontsize=7); ax.set_xlabel("t (s)", fontsize=7)
        pct = 100.0 * (t - t[0]) / (t[-1] - t[0] if t[-1] != t[0] else 1)
        self._tr_pct_line, = ax.plot(t, pct, color="#58a6ff", lw=1.2, alpha=0.4)
        self._tr_pct_pt,  = ax.plot([], [], "o", color="#f0f6fc", ms=8, zorder=5)

        # Líneas verticales animadas en los plots de tiempo
        self._tr_vl = []
        for ax in [self._ax_tr_q, self._ax_tr_qd, self._ax_tr_qdd,
                   self._ax_tr_ee, self._ax_tr_vee]:
            vl = ax.axvline(0, color=ACCENT, lw=0.8, ls=":", alpha=0.0)
            self._tr_vl.append(vl)

        # --- Vista 3D con traza completa del efector ---
        ax3 = self._ax_tr3d; ax3.clear()
        ax3.set_facecolor("#0a0f16")

        # Linea recta ideal (referencia): segmento p0 -> pf
        p0 = d["pos_ee"][0]; pf = d["pos_ee"][-1]
        ax3.plot([p0[0], pf[0]], [p0[1], pf[1]], [p0[2], pf[2]],
                 color="#8b949e", lw=1.0, ls="--", alpha=0.9,
                 label="linea recta (ideal)")

        # Trayectoria real del efector
        ax3.plot(d["pos_ee"][:, 0], d["pos_ee"][:, 1], d["pos_ee"][:, 2],
                 color="#d2a8ff", lw=1.6, alpha=0.9,
                 label=f"traza EE ({d['metodo']})")

        # Marcar inicio / fin
        ax3.scatter([p0[0]], [p0[1]], [p0[2]], color="#3fb950", s=90,
                    marker="o", edgecolors="#000", linewidth=1, label="inicio")
        ax3.scatter([pf[0]], [pf[1]], [pf[2]], color="#f85149", s=110,
                    marker="*", edgecolors="#000", linewidth=1, label="meta")
        # Dibujo inicial del brazo
        self._tr_arm_lines = []
        self._tr_arm_joints = None
        self._tr_arm_ee = None
        self._draw_arm_on(ax3, d["q"][:, 0], first=True)
        ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, loc="upper left")

        R = 400
        ax3.set_xlim(-R, R); ax3.set_ylim(-R, R); ax3.set_zlim(-R, R*0.9)
        ax3.set_xlabel("X (mm)", fontsize=8)
        ax3.set_ylabel("Y (mm)", fontsize=8)
        ax3.set_zlabel("Z (mm)", fontsize=8)
        titulo = ("Interpolacion articular: curva (FK no lineal)"
                  if d['metodo'] == "articular"
                  else "Interpolacion cartesiana: linea recta (IK en cada paso)")
        ax3.set_title(titulo, fontsize=9, color=ACCENT, pad=6)
        ax3.tick_params(labelsize=7)
        for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
            pane.set_facecolor("#0a0f16"); pane.set_edgecolor(BORDER)

        self._canvas_tr.draw_idle()

    def _draw_arm_on(self, ax3, q_deg, first=False):
        """Dibuja/actualiza los eslabones del brazo en la figura 3D de trayectoria."""
        origins = fk(q_deg)
        xs, ys, zs = origins[:, 0], origins[:, 1], origins[:, 2]

        if first:
            self._tr_arm_lines = []
            for i in range(3):
                (l,) = ax3.plot([xs[i], xs[i+1]],
                                [ys[i], ys[i+1]],
                                [zs[i], zs[i+1]],
                                color=C_J[i], lw=5, solid_capstyle="round",
                                zorder=4)
                self._tr_arm_lines.append(l)
            self._tr_arm_joints = ax3.scatter(
                xs[:-1], ys[:-1], zs[:-1], color=ACCENT,
                s=70, edgecolors="#000", linewidth=1, zorder=5)
            self._tr_arm_ee = ax3.scatter(
                [xs[-1]], [ys[-1]], [zs[-1]], color="#f78166",
                s=140, marker="*", edgecolors="#000", linewidth=1, zorder=6)
        else:
            for i, l in enumerate(self._tr_arm_lines):
                l.set_data_3d([xs[i], xs[i+1]],
                              [ys[i], ys[i+1]],
                              [zs[i], zs[i+1]])
            self._tr_arm_joints._offsets3d = (xs[:-1], ys[:-1], zs[:-1])
            self._tr_arm_ee._offsets3d     = ([xs[-1]], [ys[-1]], [zs[-1]])

    def _update_traj(self, i):
        d = self._tr_data
        if d is None:
            return
        t = d["t"]
        i = max(0, min(i, len(t) - 1))
        ti = t[i]
        pct = 100.0 * i / (len(t) - 1)

        # Brazo 3D
        self._draw_arm_on(self._ax_tr3d, d["q"][:, i], first=False)

        # Líneas verticales y marcador de progreso
        for vl in self._tr_vl:
            vl.set_xdata([ti, ti]); vl.set_alpha(0.6)
        self._tr_pct_pt.set_data([ti], [pct])

        self._lbl_status.config(
            text=f"t={ti:5.2f}s  ({pct:5.1f}%)   "
                 f"q=({d['q'][0,i]:+7.2f}, {d['q'][1,i]:+7.2f}, {d['q'][2,i]:+7.2f}) deg   "
                 f"EE=({d['pos_ee'][i,0]:+7.1f}, {d['pos_ee'][i,1]:+7.1f}, {d['pos_ee'][i,2]:+7.1f}) mm   "
                 f"|v|={d['speed_ee'][i]:6.1f} mm/s")
        self._canvas_tr.draw_idle()

    def _play_traj(self):
        if self._tr_data is None:
            self._plan_traj()
            if self._tr_data is None:
                return
        if self._tr_frame >= len(self._tr_data["t"]) - 1:
            self._tr_frame = 0
        self._tr_running = True
        self._set_play_buttons(playing=True, finished=False)
        self._anim_traj()

    def _pause_traj(self):
        self._tr_running = False
        if self._job is not None:
            self.after_cancel(self._job); self._job = None
        self._set_play_buttons(playing=False, finished=False)

    def _reset_traj(self):
        self._tr_running = False
        if self._job is not None:
            self.after_cancel(self._job); self._job = None
        self._tr_frame = 0
        self._update_traj(0)
        self._set_play_buttons(playing=False, finished=False)

    def _anim_traj(self):
        if not self._tr_running:
            return
        self._update_traj(self._tr_frame)
        self._tr_frame += 2
        if self._tr_frame >= len(self._tr_data["t"]) - 1:
            self._tr_frame = len(self._tr_data["t"]) - 1
            self._update_traj(self._tr_frame)
            self._tr_running = False
            self._set_play_buttons(playing=False, finished=True)
            return
        self._job = self.after(20, self._anim_traj)

    def _set_play_buttons(self, playing, finished):
        if playing:
            self._btn_play_tr.config(bg=BTN_DIS, fg=MUTED, state="disabled")
            self._btn_pause_tr.config(bg=BTN_PAUS, fg=ACCENT, state="normal")
        else:
            self._btn_play_tr.config(bg=BTN_PLAY, fg=ACCENT, state="normal")
            self._btn_pause_tr.config(bg=BTN_DIS, fg=MUTED, state="disabled")

    # ──────────────────────────────────────────────────────────
    def _on_close(self):
        if self._job is not None:
            try: self.after_cancel(self._job)
            except Exception: pass
        plt.close("all")
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    BrazoApp().mainloop()
