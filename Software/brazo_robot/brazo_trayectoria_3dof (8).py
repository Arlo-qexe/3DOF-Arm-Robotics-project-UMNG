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
                           Dos modos: interpolación articular (curva) o
                           cartesiana (línea recta con IK por paso).
    4)  Serial / ESP32   — Conectar con el ESP32 por serial, enviar setpoints
                           individuales por motor, hacer streaming de la
                           trayectoria planificada (respetando el perfil),
                           ver telemetría en vivo.
                           Protocolo JSON del firmware .ino:
                             {"motor":N, "pos":deg}         — un motor
                             {"m1":a, "m2":b, "m3":c}       — los tres
                             {"cmd":"reset"}                — pone pos=0
  -----------------------------------------------------------------------------
  Parámetros físicos consistentes con los códigos originales:
    DH:      a = [40, 180, 140] mm,  d = [0, 0, 0] mm,  α = [90°, 0°, 0°]
             (a1=40mm: offset horizontal del eje del primer brazo —
              gira con q1, desplazado 40mm del centro de la base)
    Gripper: L_GRIPPER = 40 mm — eslabón fijo (sin articulación) alineado
             con el eje del último eslabón. El efector final (TCP) se
             calcula en la punta del gripper; la IK recibe el TCP y resta
             el offset del gripper para hallar la posición de la muñeca.
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

import json
import time
import threading
import queue
import csv
import os
from tkinter import filedialog

# pyserial es opcional: si no esta instalado la pestaña serial avisa al usuario
try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

# rclpy es opcional: si no esta instalado la pestaña ROS2 avisa al usuario
try:
    import rclpy
    from rclpy.node import Node as RclpyNode
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    ROS2_OK = True
except ImportError:
    ROS2_OK = False

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

# Largo del efector final (gripper) — eslabón fijo sin articulación
# Se extiende desde O3 en la dirección del eje X del último sistema DH.
L_GRIPPER = 40.0    # mm

# Límites articulares (grados) — rangos físicos del brazo:
#   q1: 287° de recorrido  ([-37, +250])
#   q2: 102° de recorrido  ([+30, +132])
#   q3: 122° de recorrido  ([-130,  -8])
# Los defaults Q0_DEF=[0, 117, -118] y QF_DEF=[245, 83, -56] caen dentro.
Q_MIN = np.array([ -37.0,   30.0, -130.0])
Q_MAX = np.array([ 250.0,  132.0,   -8.0])

# Gripper MG90 (servo en el ESP32, GPIO 13).
# Convención: 0° = totalmente abierto, 180° = totalmente cerrado.
GRIP_MIN_DEG  = 0      # límite inferior del slider
GRIP_MAX_DEG  = 180    # límite superior del slider
GRIP_OPEN_DEG = 40     # posición "abierto"
GRIP_CLOSE_DEG = 130   # posición "cerrado"
GRIP_DEF_DEG  = GRIP_OPEN_DEG  # estado inicial = abierto

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
    """Cinemática directa: devuelve orígenes [O0, O1, O2, O3, TCP].

    O3  = muñeca (último eslabón DH).
    TCP = punta del gripper (L_GRIPPER mm a lo largo del eje X del marco 3).
    El gripper es un eslabón fijo (sin articulación), por lo que se modela
    como una transformación DH con theta=0, alpha=0, d=0 y a=L_GRIPPER.
    """
    q = np.deg2rad(q_deg)
    T = np.eye(4)
    origins = [T[:3, 3].copy()]
    for i in range(3):
        Ti = dh_matrix(_DH_A[i], _DH_D[i], _DH_ALPHA[i], q[i])
        T  = T @ Ti
        origins.append(T[:3, 3].copy())
    # Gripper: eslabón fijo alineado con eje X del marco 3 (theta=0, alpha=0, d=0)
    T_gripper = dh_matrix(L_GRIPPER, 0.0, 0.0, 0.0)
    T = T @ T_gripper
    origins.append(T[:3, 3].copy())   # TCP
    return np.array(origins)          # (5, 3)


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

    # ── Corrección por el gripper ─────────────────────────────────────────────
    # El TCP (x,y,z) está a L_GRIPPER mm de la muñeca O3, extendido en la
    # dirección del último eslabón. Para un robot planar (q1,q2,q3), el eje X
    # del marco 3 apunta en la dirección q2+q3 en el plano vertical y en la
    # dirección q1 en el plano horizontal.
    # Ángulo de aproximación = q1 (azimut) y ángulo de elevación = q2+q3.
    # Para descontar el gripper sin conocer q3 a priori, calculamos el vector
    # unitario desde la base hacia el target en el plano XY y restamos
    # L_GRIPPER en esa dirección (componente horizontal) y en z (vertical).
    # Esto es exacto para la solución analítica de este tipo de robot.
    r_total = np.hypot(x, y)
    if r_total > 1e-9:
        ux, uy = x / r_total, y / r_total
    else:
        ux, uy = 1.0, 0.0

    # q1: la base apunta hacia el objetivo en el plano X-Y
    q1 = np.arctan2(y, x)

    # Radio horizontal desde la base al target y su componente desde el hombro
    r_wrist_total = r_total - a1
    # El gripper se extiende horizontalmente cos(q2+q3)*L_GRIPPER y
    # verticalmente sin(q2+q3)*L_GRIPPER. Como no conocemos q2+q3 aún,
    # usamos el ángulo del vector (r_wrist_total, z) como estimación inicial.
    psi = np.arctan2(z, r_wrist_total)      # elevación al target desde el hombro
    r_w = r_wrist_total - L_GRIPPER * np.cos(psi)   # radio horizontal de la muñeca
    s_w = z             - L_GRIPPER * np.sin(psi)   # altura de la muñeca
    # ─────────────────────────────────────────────────────────────────────────

    # Verificar alcance de la muñeca
    L = np.hypot(r_w, s_w)
    if L > (L2 + L3) + 1e-9 or L < abs(L2 - L3) - 1e-9:
        return None      # fuera del espacio de trabajo

    # Ley de cosenos: ángulo del codo
    D = (r_w*r_w + s_w*s_w - L2*L2 - L3*L3) / (2.0 * L2 * L3)
    D = np.clip(D, -1.0, 1.0)
    sign = 1.0 if elbow_up else -1.0
    q3 = sign * np.arctan2(np.sqrt(1.0 - D*D), D)

    # Ángulo del hombro
    q2 = np.arctan2(s_w, r_w) - np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))

    return np.rad2deg(np.array([q1, q2, q3]))


def workspace_bounds():
    """Radios mínimo y máximo del espacio de trabajo (TCP), y altura máxima."""
    a1, L2, L3 = _DH_A
    L_total = L2 + L3 + L_GRIPPER
    r_max = a1 + L_total
    r_min = max(0.0, a1 - L_total)
    z_max =  L_total
    z_min = -L_total
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
#  JACOBIANO ANALÍTICO  (solo posición)
#
#  Para un manipulador 3-DOF con ángulos q = [q1, q2, q3] (en radianes),
#  el jacobiano analítico de posición J_A(q) es la matriz 3x3 tal que:
#
#      [ẋ]                  [q̇1]
#      [ẏ] = J_A(q) * [q̇2]
#      [ż]                  [q̇3]
#
#  Se obtiene derivando la cinemática directa de posición respecto a q:
#      x = c1 * (a1 + L2*c2 + L3*c23)
#      y = s1 * (a1 + L2*c2 + L3*c23)
#      z =       L2*s2 + L3*s23
#
#  donde c_i = cos(q_i), s_i = sin(q_i), c23 = cos(q2+q3), s23 = sin(q2+q3).
#
#  La matriz se calcula columna a columna a partir de:
#      ∂p/∂q1, ∂p/∂q2, ∂p/∂q3
# ═══════════════════════════════════════════════════════════════════════════════
def jacobiano_analitico(q_deg):
    """Devuelve el jacobiano analítico 3x3 de posición evaluado en q (deg).

    El gripper (L_GRIPPER) se modela como un aumento del último eslabón:
    L3_eff = L3 + L_GRIPPER, ya que el gripper es fijo (sin articulación) y
    colineal con el eje X del marco 3.
    """
    q = np.deg2rad(q_deg)
    q1, q2, q3 = q[0], q[1], q[2]
    a1 = _DH_A[0]
    L2 = _DH_A[1]
    L3 = _DH_A[2] + L_GRIPPER   # eslabón efectivo: antebrazo + gripper

    c1, s1   = np.cos(q1),    np.sin(q1)
    c2, s2   = np.cos(q2),    np.sin(q2)
    c23, s23 = np.cos(q2+q3), np.sin(q2+q3)

    # r(q2,q3) = a1 + L2*c2 + L3*c23   (radio horizontal del TCP)
    r = a1 + L2*c2 + L3*c23

    # Columna 1: ∂p/∂q1
    J11 = -s1 * r
    J21 =  c1 * r
    J31 =  0.0
    # Columna 2: ∂p/∂q2
    J12 = c1 * (-L2*s2 - L3*s23)
    J22 = s1 * (-L2*s2 - L3*s23)
    J32 =       L2*c2 + L3*c23
    # Columna 3: ∂p/∂q3
    J13 = c1 * (-L3*s23)
    J23 = s1 * (-L3*s23)
    J33 =       L3*c23

    J = np.array([
        [J11, J12, J13],
        [J21, J22, J23],
        [J31, J32, J33],
    ])
    return J


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTROL CINEMÁTICO CON JACOBIANO INVERSO
#
#  Esquema (ver Control_cinematico.pdf):
#
#         x_d ──►⊕──► e ──►[ K ]──┐
#                ▲                │
#               (-)               ▼
#                │   ẋ_d ─────►⊕──► [ J_A⁻¹(q) ] ─► q̇ ─►[ ∫ ] ─► q ──►...
#                │                     ▲
#                │                     │ (q realimentado)
#                │                     │
#               x_e ◄──[ k(·) ] ◄──────┘   (k(·) = cinemática directa)
#
#  Ecuaciones:
#      e   = x_d - x_e            (error en espacio operacional)
#      q̇   = J_A⁻¹(q) (ẋ_d + K·e)
#      ė + K·e = 0                (dinámica del error: convergencia exponencial)
#
#  K es la matriz de ganancias diagonal (autovalores del sistema de error).
#  Cuanto mayor K, más rápido se cancela el error de seguimiento.
#
#  Integración: Euler explícito con paso DT.
#  Si J_A se vuelve singular, se usa la pseudoinversa amortiguada (DLS).
# ═══════════════════════════════════════════════════════════════════════════════
def control_cinematico(q0_deg, xd, xd_dot, t, K_gain, dls_lambda=0.01):
    """
    Integra el controlador cinemático con jacobiano analítico.

    Parámetros
    ----------
    q0_deg   : np.ndarray (3,)
        Configuración inicial articular en grados.
    xd       : np.ndarray (N, 3)
        Posición deseada del efector final en cada instante (mm).
    xd_dot   : np.ndarray (N, 3)
        Velocidad deseada del efector final en cada instante (mm/s).
    t        : np.ndarray (N,)
        Vector de tiempo (s).
    K_gain   : float o np.ndarray (3,)
        Ganancia(s) del controlador. Escalar o vector diagonal.
    dls_lambda : float
        Factor de amortiguamiento para la pseudoinversa (DLS) cuando J
        es casi singular. λ pequeño ≈ inversa exacta; λ grande ≈ más estable.

    Devuelve
    -------
    q_deg     : (3, N) ángulos articulares en grados a lo largo del tiempo
    qd_deg    : (3, N) velocidades articulares (deg/s)
    qdd_deg   : (3, N) aceleraciones articulares (deg/s²) — por diferencias
    pos_ee    : (N, 3) posición real del efector final (mm) — vía FK
    err_ee    : (N, 3) error cartesiano  e = xd - x_e  (mm)
    """
    N  = len(t)
    DTloc = t[1] - t[0] if N > 1 else DT

    # K como matriz 3x3 diagonal
    if np.isscalar(K_gain):
        Kmat = np.eye(3) * float(K_gain)
    else:
        Kmat = np.diag(np.asarray(K_gain, dtype=float))

    q_deg  = np.zeros((3, N))
    qd_deg = np.zeros((3, N))
    pos_ee = np.zeros((N, 3))
    err_ee = np.zeros((N, 3))

    # Estado inicial
    q_deg[:, 0]   = q0_deg
    pos_ee[0]     = ee_pos(q0_deg)
    err_ee[0]     = xd[0] - pos_ee[0]

    for k in range(N - 1):
        q_k = q_deg[:, k]
        # Cinemática directa (k(·))
        x_e = ee_pos(q_k)
        # Error en espacio operacional
        e = xd[k] - x_e
        # Velocidad deseada de articulación: q̇ = J⁻¹(q) (ẋ_d + K·e)
        v_des = xd_dot[k] + Kmat @ e
        J = jacobiano_analitico(q_k)

        # Inversa o pseudoinversa amortiguada (DLS) por estabilidad numérica
        try:
            # Si el determinante es razonable usamos inversa directa
            detJ = np.linalg.det(J)
            if abs(detJ) > 1e-3:
                qd_rad = np.linalg.solve(J, v_des)
            else:
                # DLS:  qd = Jᵀ (J Jᵀ + λ²I)⁻¹ v_des
                JJt   = J @ J.T + (dls_lambda**2) * np.eye(3)
                qd_rad = J.T @ np.linalg.solve(JJt, v_des)
        except np.linalg.LinAlgError:
            JJt   = J @ J.T + (dls_lambda**2) * np.eye(3)
            qd_rad = J.T @ np.linalg.solve(JJt, v_des)

        # Las posiciones articulares se manejan en grados, así que pasamos q̇
        # a deg/s antes de integrar
        qd_dps = np.rad2deg(qd_rad)
        qd_deg[:, k]   = qd_dps

        # Euler explícito: q(k+1) = q(k) + q̇·dt
        q_deg[:, k+1]  = q_k + qd_dps * DTloc

        pos_ee[k+1]    = ee_pos(q_deg[:, k+1])
        err_ee[k+1]    = xd[k+1] - pos_ee[k+1]

    # Última velocidad (mismo q̇ que el paso anterior, no se calcula otro)
    qd_deg[:, -1] = qd_deg[:, -2] if N > 1 else 0.0

    # Aceleración por diferencias
    qdd_deg = np.gradient(qd_deg, t, axis=1)

    return q_deg, qd_deg, qdd_deg, pos_ee, err_ee


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
#  WrappingFrame — barra de controles que se envuelve a múltiples filas
#
#  Tkinter no tiene un layout "flow" nativo. Esta clase implementa uno simple:
#  los widgets agregados con .add() se colocan de izquierda a derecha y, cuando
#  no cabe el siguiente en el ancho disponible, baja a la siguiente fila.
#  El reflow se dispara automáticamente con <Configure> (cambios de tamaño).
#
#  Uso:
#     bar = WrappingFrame(parent, bg=PANEL, hgap=6, vgap=4)
#     bar.pack(fill="x")
#     bar.add(tk.Button(bar, text="Algo"))   # ← parent debe ser bar
#     bar.add(tk.Label(bar, text="..."))
#     bar.finish()                           # llamar tras agregar todos
# ═══════════════════════════════════════════════════════════════════════════════
class WrappingFrame(tk.Frame):
    def __init__(self, parent, hgap=6, vgap=4, **kw):
        super().__init__(parent, **kw)
        self._children = []         # widgets en el orden agregado
        self._hgap = hgap
        self._vgap = vgap
        self._after_id = None
        self.bind("<Configure>", self._on_configure)

    def add(self, w, padx=None, pady=None):
        self._children.append((w, padx if padx is not None else self._hgap,
                                  pady if pady is not None else self._vgap))
        return w

    def finish(self):
        self.update_idletasks()
        self._reflow()

    def _on_configure(self, _evt):
        # Throttle: reagrupar tras un pequeño retardo para evitar parpadeos
        if self._after_id is not None:
            try: self.after_cancel(self._after_id)
            except Exception: pass
        self._after_id = self.after(15, self._reflow)

    def _reflow(self):
        self._after_id = None
        avail_w = self.winfo_width()
        if avail_w < 10:
            return  # aun no se ha asignado tamaño

        x = 0
        y = 0
        row_h = 0
        for w, padx, pady in self._children:
            try:
                w.update_idletasks()
                wreq = w.winfo_reqwidth()
                hreq = w.winfo_reqheight()
            except tk.TclError:
                continue
            # Si no cabe, salto de fila
            if x + wreq + padx > avail_w and x > 0:
                x = 0
                y += row_h + self._vgap
                row_h = 0
            try:
                w.place(in_=self, x=x + padx // 2, y=y + pady // 2)
            except tk.TclError:
                continue
            x += wreq + padx
            if hreq > row_h:
                row_h = hreq
        # Ajustar la altura del frame para que ocupe todas las filas
        total_h = y + row_h + self._vgap
        self.configure(height=max(total_h, 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  ScrollableFrame — frame interno con scroll vertical y horizontal opcional.
#  Se usa como red de seguridad: cuando la ventana es muy estrecha y los
#  controles no caben aun envolviendo, el usuario puede desplazarse.
# ═══════════════════════════════════════════════════════════════════════════════
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg=PANEL, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._sb_v = tk.Scrollbar(self, orient="vertical",
                                  command=self._canvas.yview)
        self._sb_v.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=self._sb_v.set)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self._win = self._canvas.create_window((0, 0), window=self.inner,
                                               anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_canvas_resize(self, evt):
        # Hacer que el frame interno tenga al menos el ancho del canvas para
        # que los hijos con fill="x" funcionen.
        self._canvas.itemconfigure(self._win, width=evt.width)


# ═══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class BrazoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Brazo Robótico 3-DOF — Simulador Integrado")
        self.configure(bg=BG)

        # Tamaño inicial responsive: usa el tamaño de la pantalla pero con
        # un techo razonable para que no salga gigantesco en monitores 4K.
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w0 = min(int(sw * 0.92), 1600)
        h0 = min(int(sh * 0.88),  950)
        # Centrar
        x0 = (sw - w0) // 2
        y0 = (sh - h0) // 2
        self.geometry(f"{w0}x{h0}+{x0}+{y0}")
        # Tamaño minimo: si la pantalla es pequeña la app sigue siendo usable.
        # Por debajo de esto los controles empiezan a ser inutilizables.
        self.minsize(900, 600)
        # Intentar maximizar (si la plataforma lo soporta), sin fallar si no.
        try:
            self.state("zoomed")              # Windows
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)   # algunos Linux
            except tk.TclError:
                pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._tab = 0
        self._job = None

        # Estado articular actual (se usa en la pestaña 3D y como q_inicio en trayectoria)
        self._q_current = Q0_DEF.copy()

        # ── Grabación de trayectoria manual (pestaña 1) ──
        self._recording = False           # True mientras se graba
        self._rec_keyframes = []          # lista de np.array(q) — poses capturadas
        self._rec_times     = []          # tiempos relativos de cada keyframe (s)
        self._rec_t_start   = 0.0         # tiempo monotónico al iniciar

        # ── Estado serial / ESP32 ──
        self._serial = None                      # objeto serial.Serial (o None)
        self._serial_rx_queue = queue.Queue()    # cola de lineas recibidas (hilo -> UI)
        self._serial_rx_thread = None
        self._serial_rx_stop = threading.Event()
        self._serial_pump_job = None             # after() del drenado de la cola
        # Ultima telemetria parseada: dict con m1/m2/m3 -> (pos_deg, ref_deg, ok)
        self._tel = {f"m{j}": {"pos": 0.0, "ref": 0.0, "ok": True} for j in (1,2,3)}
        # Estado del gripper (servo MG90) — pos en grados [0..180], 0=abierto
        self._tel["grip"] = {"pos": GRIP_DEF_DEG, "abierto": True}

        # Historial de telemetria para graficar (deque manual con numpy)
        self._tel_max = 400
        self._tel_t   = np.zeros(self._tel_max)
        self._tel_pos = np.zeros((3, self._tel_max))
        self._tel_ref = np.zeros((3, self._tel_max))
        self._tel_idx = 0
        self._tel_t0  = time.monotonic()

        # Streaming de trayectoria
        self._stream_thread = None
        self._stream_stop = threading.Event()

        # ── ROS2 / Gazebo bridge ──
        self._ros2_node = None
        self._ros2_thread = None
        self._ros2_pub = None
        self._ros2_running = False
        self._ros2_auto_pub = False
        self._ros2_last_cmd = None          # ultimo Float64MultiArray enviado (republica a 20 Hz)
        self._ros2_stream_thread = None
        self._ros2_stream_stop = threading.Event()
        self._ros2_feedback_q = [0.0, 0.0, 0.0]  # rad (desde /joint_states)
        self._ros2_feedback_labels = []            # Labels de la UI (se crean en _build_tab_ros2)

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
        bar_outer = tk.Frame(self, bg=PANEL, pady=2)
        bar_outer.pack(fill="x", side="top")
        bar = WrappingFrame(bar_outer, bg=PANEL, hgap=3, vgap=3)
        bar.pack(fill="x", padx=4)
        self._tab_btns = []
        # Etiquetas mas cortas para que quepan mejor en pantallas estrechas
        labels = ["  1. Vista 3D  ", "  2. Motores + PID  ",
                  "  3. Trayectoria  ",
                  "  4. Serial / ESP32  ",
                  "  5. Ping pong  ",
                  "  6. ROS2 / Gazebo  "]
        for i, lbl in enumerate(labels):
            b = tk.Button(bar, text=lbl,
                          bg=BTN_HOV if i == 0 else BTN_BG,
                          fg=ACCENT if i == 0 else TEXT,
                          relief="flat", bd=0,
                          font=("Courier New", 10, "bold"),
                          cursor="hand2", padx=12, pady=6,
                          command=lambda i=i: self._switch_tab(i))
            bar.add(b)
            self._tab_btns.append(b)
        bar.finish()
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
        self._frame_3d       = tk.Frame(self._main, bg=BG)
        self._frame_pid      = tk.Frame(self._main, bg=BG)
        self._frame_traj     = tk.Frame(self._main, bg=BG)
        self._frame_serial   = tk.Frame(self._main, bg=BG)
        self._frame_pingpong = tk.Frame(self._main, bg=BG)
        self._frame_ros2     = tk.Frame(self._main, bg=BG)

        self._build_tab_3d(self._frame_3d)
        self._build_tab_pid(self._frame_pid)
        self._build_tab_traj(self._frame_traj)
        self._build_tab_serial(self._frame_serial)
        self._build_tab_pingpong(self._frame_pingpong)
        self._build_tab_ros2(self._frame_ros2)

    def _switch_tab(self, idx):
        # Parar cualquier animación pendiente de otras pestañas
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

        # Si el modo JOG estaba activo y nos vamos de la pestaña Serial,
        # lo desactivamos por seguridad (sino las teclas a/d/w/s/q/e/p
        # seguirian moviendo el robot mientras el usuario escribe en otro
        # campo o navega).
        if (idx != 3) and getattr(self, "_jog_active", False):
            try:
                self._ctrl_mode.set("sliders")
                self._jog_disable()
            except Exception:
                pass

        for i, b in enumerate(self._tab_btns):
            if i == idx:
                b.config(bg=BTN_HOV, fg=ACCENT)
            else:
                b.config(bg=BTN_BG, fg=TEXT)

        for f in (self._frame_3d, self._frame_pid, self._frame_traj,
                  self._frame_serial, self._frame_pingpong, self._frame_ros2):
            f.pack_forget()
        target = [self._frame_3d, self._frame_pid, self._frame_traj,
                  self._frame_serial, self._frame_pingpong, self._frame_ros2][idx]
        target.pack(fill="both", expand=True)
        self._tab = idx

    # ═══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA 1 — VISTA 3D
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_tab_3d(self, parent):
        # Contenedor de los dos paneles inferiores (se apilan, el último en poner
        # ocupa el fondo → pack side="bottom" en orden inverso)
        bottom = tk.Frame(parent, bg=PANEL)
        bottom.pack(fill="x", side="bottom")

        tk.Frame(bottom, bg=BORDER, height=1).pack(fill="x")   # separador

        # ── FILA 2: controles de grabación (envuelve a varias filas) ──
        row2_outer = tk.Frame(bottom, bg="#0d1a0d", pady=4)
        row2_outer.pack(fill="x", side="bottom")
        row2 = WrappingFrame(row2_outer, bg="#0d1a0d", hgap=4, vgap=3)
        row2.pack(fill="x", padx=10)

        row2.add(tk.Label(row2, text="GRABAR MOVIMIENTO:",
                          bg="#0d1a0d", fg="#f0c040",
                          font=("Courier New", 10, "bold")))

        self._btn_rec = tk.Button(
            row2, text="  ⏺ Iniciar grabación  ",
            bg="#b62324", fg=ACCENT,
            relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=10, pady=4,
            command=self._toggle_recording)
        row2.add(self._btn_rec)

        row2.add(tk.Button(row2, text="  + Agregar pose  ",
                  bg=BTN_HOV, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=10, pady=4,
                  command=self._add_keyframe))

        row2.add(tk.Button(row2, text="  ✖ Limpiar  ",
                  bg=BTN_BG, fg=MUTED,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=10, pady=4,
                  command=self._clear_recording))

        row2.add(tk.Button(row2, text="  💾 Generar CSV  ",
                  bg=BTN_PLAY, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=10, pady=4,
                  command=self._export_csv))

        row2.add(tk.Button(row2, text="  📂 Importar CSV  ",
                  bg=BTN_RSET, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=10, pady=4,
                  command=self._import_csv))

        frm_dur, self._rec_dur_var = make_labeled_entry(row2, "seg/tramo", 2.0, width=5)
        row2.add(frm_dur)

        row2.add(tk.Frame(row2, bg="#1a2a1a", width=2, height=24))

        self._lbl_rec_status = tk.Label(
            row2, text="Sin grabación activa — presiona ⏺ para comenzar",
            bg="#0d1a0d", fg="#5a8a5a",
            font=("Courier New", 9))
        row2.add(self._lbl_rec_status)
        row2.finish()

        tk.Frame(bottom, bg=BORDER, height=1).pack(fill="x")   # separador

        # ── FILA 1: sliders + botones de pose (envuelve) ──
        row1_outer = tk.Frame(bottom, bg=PANEL, pady=6)
        row1_outer.pack(fill="x", side="bottom")
        row1 = WrappingFrame(row1_outer, bg=PANEL, hgap=6, vgap=4)
        row1.pack(fill="x", padx=10)

        row1.add(tk.Label(row1, text="Angulos articulares (deg)",
                          bg=PANEL, fg=ACCENT,
                          font=("Courier New", 10, "bold")))

        self._sliders_3d = []
        self._lbl_slider = []
        for j in range(3):
            col = tk.Frame(row1, bg=PANEL)
            tk.Label(col, text=JNAMES[j], bg=PANEL,
                     fg=C_J[j], font=("Courier New", 10, "bold")).pack()
            var = tk.DoubleVar(value=self._q_current[j])
            s = tk.Scale(col, from_=Q_MIN[j], to=Q_MAX[j],
                         resolution=1.0, orient="horizontal",
                         length=180, bg=PANEL, fg=TEXT,
                         troughcolor=BTN_BG, highlightthickness=0,
                         font=("Courier New", 8),
                         variable=var,
                         command=lambda _v, j=j: self._on_slider_3d(j))
            s.pack()
            row1.add(col)
            self._sliders_3d.append(var)

        row1.add(tk.Frame(row1, bg=BORDER, width=2, height=40))

        row1.add(tk.Button(row1, text="  Pose inicial  ",
                  bg=BTN_RSET, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=12, pady=5,
                  command=self._reset_pose_3d))

        row1.add(tk.Button(row1, text="  Cero (q=0)  ",
                  bg=BTN_BG, fg=TEXT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=12, pady=5,
                  command=self._zero_pose_3d))

        # Info efector final
        self._lbl_ee = tk.Label(row1, text="", bg=PANEL, fg=ACCENT,
                                font=("Courier New", 10, "bold"))
        row1.add(self._lbl_ee)
        row1.finish()

        # Figura 3D
        self._fig3d = plt.Figure(figsize=(8, 5), facecolor=BG)
        self._ax3d  = self._fig3d.add_subplot(111, projection="3d")
        self._fig3d.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

        self._canvas3d = FigureCanvasTkAgg(self._fig3d, master=parent)
        self._canvas3d.get_tk_widget().pack(fill="both", expand=True, side="top")

        self._draw_3d(self._q_current)

    def _on_slider_3d(self, j):
        self._q_current[j] = self._sliders_3d[j].get()
        self._draw_3d(self._q_current)
        if self._ros2_auto_pub:
            self._ros2_publish_q(self._q_current)

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

    # ──────────────────────────────────────────
    #  GRABACIÓN DE TRAYECTORIA MANUAL
    # ──────────────────────────────────────────
    def _toggle_recording(self):
        if not self._recording:
            # Iniciar grabación
            self._recording = True
            self._rec_keyframes = [self._q_current.copy()]
            self._rec_times     = [0.0]
            self._rec_t_start   = time.monotonic()
            self._btn_rec.config(text="  ⏹ Detener grabación  ", bg=BTN_PAUS)
            self._lbl_rec_status.config(
                text=f"● GRABANDO — {len(self._rec_keyframes)} pose(s) | Mueve el brazo y presiona '+ Agregar pose'")
            self._lbl_status.config(
                text="Grabación iniciada. Mueve los sliders y presiona '+ Agregar pose' para capturar waypoints.")
        else:
            # Detener grabación
            self._recording = False
            n = len(self._rec_keyframes)
            self._btn_rec.config(text="  ⏺ Iniciar grabación  ", bg="#b62324")
            self._lbl_rec_status.config(
                text=f"Grabación detenida — {n} pose(s) registradas. Presiona '💾 Generar CSV' para exportar.")
            self._lbl_status.config(
                text=f"Grabación detenida. {n} keyframes guardados. Usa '💾 Generar CSV' para generar la trayectoria.")

    def _add_keyframe(self):
        if not self._recording:
            messagebox.showwarning("Sin grabación activa",
                                   "Primero presiona '⏺ Iniciar grabación' para comenzar.")
            return
        t_rel = time.monotonic() - self._rec_t_start
        q_snap = self._q_current.copy()
        self._rec_keyframes.append(q_snap)
        self._rec_times.append(t_rel)
        n = len(self._rec_keyframes)
        self._lbl_rec_status.config(
            text=f"● GRABANDO — {n} pose(s) guardadas   "
                 f"| última: q=({q_snap[0]:+.1f}, {q_snap[1]:+.1f}, {q_snap[2]:+.1f})°  t={t_rel:.2f}s")
        self._lbl_status.config(
            text=f"Keyframe #{n} capturado en t={t_rel:.2f}s  "
                 f"— q=({q_snap[0]:+.1f}°, {q_snap[1]:+.1f}°, {q_snap[2]:+.1f}°)")

    def _clear_recording(self):
        self._recording = False
        self._rec_keyframes = []
        self._rec_times     = []
        self._btn_rec.config(text="  ⏺ Iniciar grabación  ", bg="#b62324")
        self._lbl_rec_status.config(text="")
        self._lbl_status.config(text="Grabación limpiada.")

    def _export_csv(self):
        """
        Genera la trayectoria quíntica completa entre los keyframes grabados
        y la exporta como CSV con columnas:
            t, q1_deg, q2_deg, q3_deg, q1d_dps, q2d_dps, q3d_dps,
            q1dd_dps2, q2dd_dps2, q3dd_dps2, ee_x_mm, ee_y_mm, ee_z_mm
        """
        if len(self._rec_keyframes) < 2:
            messagebox.showerror(
                "Pocas poses",
                "Necesitas al menos 2 poses para generar una trayectoria.\n\n"
                "Pasos:\n  1. Presiona '⏺ Iniciar grabación'\n"
                "  2. Mueve los sliders a la pose deseada\n"
                "  3. Presiona '+ Agregar pose' (repite para más puntos)\n"
                "  4. Presiona '💾 Generar CSV'")
            return

        # Duración por tramo
        try:
            tf_seg = float(self._rec_dur_var.get())
            if tf_seg <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Duración inválida", "Ingresa un valor positivo en 'seg/tramo'.")
            return

        # Construir trayectoria quíntica tramo a tramo
        t_global = []
        q_global  = [[], [], []]
        qd_global = [[], [], []]
        qdd_global= [[], [], []]
        t_offset = 0.0

        for seg in range(len(self._rec_keyframes) - 1):
            q0 = self._rec_keyframes[seg]
            qf = self._rec_keyframes[seg + 1]
            t_local = np.arange(0.0, tf_seg + DT, DT)
            # Evitar duplicar el punto de unión entre tramos
            if seg > 0:
                t_local = t_local[1:]
            q, qd, qdd = perfil_quintico(q0, qf, t_local, tf_seg)
            for j in range(3):
                q_global[j].extend(q[j].tolist())
                qd_global[j].extend(qd[j].tolist())
                qdd_global[j].extend(qdd[j].tolist())
            t_global.extend((t_local + t_offset).tolist())
            t_offset += tf_seg

        t_arr  = np.array(t_global)
        q_arr  = np.array(q_global)    # (3, N)
        qd_arr = np.array(qd_global)
        qdd_arr= np.array(qdd_global)

        # Cinemática directa para cada punto → posición del efector final
        N = len(t_arr)
        ee_arr = np.array([ee_pos(q_arr[:, i]) for i in range(N)])

        # Diálogo para guardar
        default_name = "trayectoria_brazo.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name,
            title="Guardar trayectoria CSV")
        if not filepath:
            return  # usuario canceló

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_s",
                "q1_deg", "q2_deg", "q3_deg",
                "q1d_dps", "q2d_dps", "q3d_dps",
                "q1dd_dps2", "q2dd_dps2", "q3dd_dps2",
                "ee_x_mm", "ee_y_mm", "ee_z_mm"
            ])
            for i in range(N):
                writer.writerow([
                    f"{t_arr[i]:.4f}",
                    f"{q_arr[0,i]:.4f}", f"{q_arr[1,i]:.4f}", f"{q_arr[2,i]:.4f}",
                    f"{qd_arr[0,i]:.4f}", f"{qd_arr[1,i]:.4f}", f"{qd_arr[2,i]:.4f}",
                    f"{qdd_arr[0,i]:.4f}", f"{qdd_arr[1,i]:.4f}", f"{qdd_arr[2,i]:.4f}",
                    f"{ee_arr[i,0]:.4f}", f"{ee_arr[i,1]:.4f}", f"{ee_arr[i,2]:.4f}",
                ])

        # Guardar también en _tr_data para que la pestaña 3 pueda animarlo
        self._tr_data = dict(
            t=t_arr, q=q_arr, qd=qd_arr, qdd=qdd_arr,
            q0=self._rec_keyframes[0], qf=self._rec_keyframes[-1],
            tf=t_arr[-1],
            p0=ee_arr[0], pf=ee_arr[-1],
            pos_ee=ee_arr,
            v_ee=np.gradient(ee_arr, t_arr, axis=0),
            speed_ee=np.linalg.norm(np.gradient(ee_arr, t_arr, axis=0), axis=1),
            metodo="manual (keyframes)",
        )

        n_keyframes = len(self._rec_keyframes)
        n_pts = N
        tf_total = t_arr[-1]
        messagebox.showinfo(
            "CSV generado",
            f"Trayectoria exportada correctamente.\n\n"
            f"  Archivo : {os.path.basename(filepath)}\n"
            f"  Keyframes: {n_keyframes}\n"
            f"  Tramos   : {n_keyframes - 1}  ({tf_seg:.2f} s/tramo)\n"
            f"  Tiempo total: {tf_total:.2f} s\n"
            f"  Puntos totales (dt={DT}s): {n_pts}\n\n"
            f"Columnas: t, q1/q2/q3 (deg), velocidades (deg/s),\n"
            f"          aceleraciones (deg/s²), EE x/y/z (mm)\n\n"
            f"La trayectoria también queda cargada en la\n"
            f"pestaña '3. Trayectoria' para animarla.")
        self._lbl_status.config(
            text=f"✅ CSV guardado: {os.path.basename(filepath)}  "
                 f"— {n_keyframes} keyframes, {n_pts} puntos, tf={tf_total:.2f}s")
        # Actualizar la gráfica de la pestaña 3
        self._switch_tab(2)
        self._draw_traj_static()
        self._update_traj(0)

    def _import_csv(self):
        """Lee un CSV exportado previamente y carga la trayectoria en la pestaña 3.
        Columnas esperadas (mismo formato que _export_csv / _pingpong_generate_csv):
            t_s, q1_deg, q2_deg, q3_deg,
            q1d_dps, q2d_dps, q3d_dps,
            q1dd_dps2, q2dd_dps2, q3dd_dps2,
            ee_x_mm, ee_y_mm, ee_z_mm
        """
        filepath = filedialog.askopenfilename(
            title="Importar trayectoria CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filepath:
            return

        try:
            rows = []
            with open(filepath, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)   # saltar cabecera
                for row in reader:
                    stripped = [c.strip() for c in row]
                    if stripped and stripped[0]:
                        rows.append([float(x) for x in stripped])

            if not rows:
                messagebox.showerror("CSV vacío",
                                     "El archivo no contiene filas de datos.")
                return

            data = np.array(rows)
            if data.shape[1] < 13:
                messagebox.showerror(
                    "Formato incorrecto",
                    f"Se esperan al menos 13 columnas:\n"
                    f"  t_s, q1-q3 (deg), q1d-q3d (dps),\n"
                    f"  q1dd-q3dd (dps²), ee_x/y/z (mm)\n\n"
                    f"El archivo tiene {data.shape[1]} columna(s).")
                return

            t       = data[:, 0]
            q       = data[:, 1:4].T     # (3, N) grados
            qd      = data[:, 4:7].T     # (3, N) deg/s
            qdd     = data[:, 7:10].T    # (3, N) deg/s²
            pos_ee  = data[:, 10:13]     # (N, 3) mm
            v_ee    = np.gradient(pos_ee, t, axis=0)
            speed_ee = np.linalg.norm(v_ee, axis=1)

            self._tr_data = dict(
                t=t, q=q, qd=qd, qdd=qdd,
                q0=q[:, 0].copy(), qf=q[:, -1].copy(),
                tf=float(t[-1]),
                p0=pos_ee[0].copy(), pf=pos_ee[-1].copy(),
                pos_ee=pos_ee,
                v_ee=v_ee, speed_ee=speed_ee,
                metodo=f"importado ({os.path.basename(filepath)})",
                err_ee=None, K_used=None, xd_track=None,
            )
            self._tr_frame = 0

            N   = len(t)
            tf  = float(t[-1])
            self._switch_tab(2)
            self._draw_traj_static()
            self._update_traj(0)
            self._set_play_buttons(playing=False, finished=False)

            self._lbl_status.config(
                text=f"✅ CSV importado: {os.path.basename(filepath)}  "
                     f"— {N} puntos, tf={tf:.2f}s  "
                     f"| q0=({q[0,0]:.1f}, {q[1,0]:.1f}, {q[2,0]:.1f})°  "
                     f"→ qf=({q[0,-1]:.1f}, {q[1,-1]:.1f}, {q[2,-1]:.1f})°")

        except Exception as exc:
            messagebox.showerror("Error al importar CSV", str(exc))

    def _draw_3d(self, q_deg):
        ax = self._ax3d
        ax.clear()
        ax.set_facecolor("#0a0f16")

        origins = fk(q_deg)                  # (5,3): O0,O1,O2,O3,TCP
        xs, ys, zs = origins[:, 0], origins[:, 1], origins[:, 2]

        # Suelo (malla tenue)
        R = 350
        ax.plot([-R, R], [0, 0], [0, 0], color="#30363d", lw=0.6)
        ax.plot([0, 0], [-R, R], [0, 0], color="#30363d", lw=0.6)

        # Eslabones articulados (O0→O1→O2→O3)
        link_colors = ["#8b949e", "#58a6ff", "#3fb950", "#f78166"]
        ax.plot(xs[:4], ys[:4], zs[:4], color="#8b949e", lw=1.0, alpha=0.4)  # traza guía
        for i in range(3):
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [zs[i], zs[i+1]],
                    color=C_J[i], lw=5, solid_capstyle="round")

        # Gripper (O3→TCP) — línea diferenciada (naranja claro, más delgada)
        ax.plot([xs[3], xs[4]], [ys[3], ys[4]], [zs[3], zs[4]],
                color="#ffa657", lw=3, linestyle="--", solid_capstyle="round",
                label=f"gripper ({L_GRIPPER:.0f} mm)")

        # Articulaciones (O0 a O3)
        ax.scatter(xs[:4], ys[:4], zs[:4], color=ACCENT,
                   s=70, edgecolors="#000", linewidth=1, zorder=5)
        # TCP (punta del gripper)
        ax.scatter([xs[4]], [ys[4]], [zs[4]], color="#ffa657",
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

        # Mostrar traza de keyframes grabados (si hay alguno)
        if len(self._rec_keyframes) >= 1:
            kf_ee = np.array([ee_pos(q) for q in self._rec_keyframes])
            ax.scatter(kf_ee[:, 0], kf_ee[:, 1], kf_ee[:, 2],
                       color="#f0c040", s=90, marker="D",
                       edgecolors="#000", linewidth=1, zorder=8,
                       label=f"keyframes ({len(self._rec_keyframes)})")
            if len(self._rec_keyframes) >= 2:
                ax.plot(kf_ee[:, 0], kf_ee[:, 1], kf_ee[:, 2],
                        color="#f0c040", lw=1.2, ls="--", alpha=0.7)
            ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER,
                      loc="upper left")

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
        self._fig_pid = plt.Figure(figsize=(9, 5.5), facecolor=BG)
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

        # ── Fila 1: selector de modo + entradas (envoltura automatica) ──
        row1 = WrappingFrame(ctl, bg=PANEL, hgap=4, vgap=3)
        row1.pack(fill="x", pady=(2, 4), padx=4)

        row1.add(tk.Label(row1, text="Meta:",
                          bg=PANEL, fg=ACCENT,
                          font=("Courier New", 10, "bold")))
        self._tr_mode = tk.StringVar(value="articular")
        for lbl, val in [("Articular (q1,q2,q3)", "articular"),
                         ("Cartesiano (X,Y,Z)", "cartesiano")]:
            row1.add(tk.Radiobutton(row1, text=lbl, variable=self._tr_mode, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2",
                           command=self._update_traj_entry_visibility))

        row1.add(tk.Frame(row1, bg=BORDER, width=2, height=26))

        # Entradas articulares (meta) — se agregan/quitan dinamicamente
        self._frm_art = tk.Frame(row1, bg=PANEL)
        tk.Label(self._frm_art, text="qf (deg):",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(0, 4))
        self._tr_qf = []
        for j in range(3):
            frm, var = make_labeled_entry(self._frm_art, JNAMES[j], QF_DEF[j])
            frm.pack(side="left", padx=3); self._tr_qf.append(var)

        # Entradas cartesianas (meta)  — se agregan/quitan dinamicamente
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
        # Guardar referencia al WrappingFrame de fila 1 para actualizar visibilidad
        self._tr_row1 = row1
        # El meta articular/cartesiano se inserta como ultimo widget; lo registramos
        # como un slot reutilizable
        self._tr_meta_slot = None
        # Inicializar con articular
        row1.add(self._frm_art)
        self._tr_meta_slot = self._frm_art

        # ── Fila 2: q inicial + metodo + tf + botones (tambien envuelve) ──
        row2 = WrappingFrame(ctl, bg=PANEL, hgap=4, vgap=3)
        row2.pack(fill="x", pady=(2, 2), padx=4)
        self._tr_row2 = row2

        row2.add(tk.Label(row2, text="q inicial (deg):",
                          bg=PANEL, fg=MUTED,
                          font=("Courier New", 9)))
        self._tr_q0 = []
        for j in range(3):
            frm, var = make_labeled_entry(row2, JNAMES[j], Q0_DEF[j])
            row2.add(frm)
            self._tr_q0.append(var)

        row2.add(tk.Frame(row2, bg=BORDER, width=2, height=26))

        row2.add(tk.Label(row2, text="Interp:",
                          bg=PANEL, fg=ACCENT,
                          font=("Courier New", 9, "bold")))
        self._tr_interp = tk.StringVar(value="articular")
        for lbl, val in [("Articular (curva)", "articular"),
                         ("Cartesiano (recta)", "cartesiano"),
                         ("Control cinematico (Jacobiano)", "ctrl_cinematico")]:
            row2.add(tk.Radiobutton(row2, text=lbl, variable=self._tr_interp, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2"))

        row2.add(tk.Frame(row2, bg=BORDER, width=2, height=26))

        frm, self._tr_tf = make_labeled_entry(row2, "tf(s)", T_TOTAL_DEF, width=5)
        row2.add(frm)

        # Ganancia K del control cinematico (matriz diagonal K*I)
        # Solo se usa cuando Interp = "Control cinematico (Jacobiano)"
        frm, self._tr_K = make_labeled_entry(row2, "K", "10.0", width=5)
        row2.add(frm)

        row2.add(tk.Frame(row2, bg=BORDER, width=2, height=26))

        # ── Gripper en trayectoria ──
        self._tr_grip_en = tk.BooleanVar(value=False)
        row2.add(tk.Checkbutton(row2, text="Grip", variable=self._tr_grip_en,
                                bg=PANEL, fg=ACCENT, selectcolor=BTN_HOV,
                                activebackground=PANEL, activeforeground=ACCENT,
                                font=("Courier New", 9, "bold"), cursor="hand2",
                                command=self._update_grip_traj_visibility))

        self._frm_grip_traj = tk.Frame(row2, bg=PANEL)
        self._tr_grip_start = tk.StringVar(value="abrir")
        self._tr_grip_end   = tk.StringVar(value="cerrar")
        tk.Label(self._frm_grip_traj, text="inicio:",
                 bg=PANEL, fg=MUTED, font=("Courier New", 9)).pack(side="left")
        for lbl, val in [("◯", "abrir"), ("⬤", "cerrar")]:
            tk.Radiobutton(self._frm_grip_traj, text=lbl, variable=self._tr_grip_start,
                           value=val, bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2").pack(side="left")
        tk.Label(self._frm_grip_traj, text="  fin:",
                 bg=PANEL, fg=MUTED, font=("Courier New", 9)).pack(side="left")
        for lbl, val in [("◯", "abrir"), ("⬤", "cerrar")]:
            tk.Radiobutton(self._frm_grip_traj, text=lbl, variable=self._tr_grip_end,
                           value=val, bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2").pack(side="left")

        row2.add(tk.Button(row2, text="  Planificar  ",
                  bg=BTN_HOV, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=12, pady=6,
                  command=self._plan_traj))

        self._btn_play_tr = tk.Button(
            row2, text="  ▶ Animar  ",
            bg=BTN_PLAY, fg=ACCENT,
            relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=12, pady=6,
            command=self._play_traj)
        row2.add(self._btn_play_tr)

        self._btn_pause_tr = tk.Button(
            row2, text="  ⏸ Pausar  ",
            bg=BTN_DIS, fg=MUTED, state="disabled",
            relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=12, pady=6,
            command=self._pause_traj)
        row2.add(self._btn_pause_tr)

        row2.add(tk.Button(row2, text="  ↺ Reset  ",
                  bg=BTN_RSET, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=12, pady=6,
                  command=self._reset_traj))

        row2.add(tk.Button(row2, text="  📂 Importar CSV  ",
                  bg=BTN_BG, fg=TEXT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=12, pady=6,
                  command=self._import_csv))

        row1.finish()
        row2.finish()

        # Figura: izquierda 3D, derecha q/qd/qdd + EE
        self._fig_tr = plt.Figure(figsize=(9, 5.5), facecolor=BG)
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
        """Muestra/oculta las entradas de articular o cartesiano segun el modo.
        Trabaja con WrappingFrame: el sub-frame activo se mantiene en la lista
        de hijos y se reflowea."""
        target = (self._frm_art if self._tr_mode.get() == "articular"
                  else self._frm_cart)
        if self._tr_meta_slot is target:
            return  # ya estaba activo
        # Reemplazar en la lista del WrappingFrame
        if self._tr_meta_slot is not None:
            try:
                self._tr_meta_slot.place_forget()
            except tk.TclError:
                pass
            # quitarlo de la lista de hijos del wrapper
            self._tr_row1._children = [
                (w, px, py) for (w, px, py) in self._tr_row1._children
                if w is not self._tr_meta_slot
            ]
        self._tr_row1.add(target)
        self._tr_meta_slot = target
        self._tr_row1.finish()

    def _update_grip_traj_visibility(self):
        if self._tr_grip_en.get():
            self._tr_row2.add(self._frm_grip_traj)
        else:
            try:
                self._frm_grip_traj.place_forget()
            except tk.TclError:
                pass
            self._tr_row2._children = [
                (w, px, py) for (w, px, py) in self._tr_row2._children
                if w is not self._frm_grip_traj
            ]
        self._tr_row2.finish()

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

        # Información extra (algunos métodos la sobre-escriben)
        err_ee     = None      # error cartesiano e = xd - x_e (solo control cin.)
        K_used     = None      # ganancia K usada (solo control cin.)
        xd_track   = None      # trayectoria deseada (solo control cin., para graficar)

        if metodo == "articular":
            # --- Interpolar los angulos articulares con perfil quintico ---
            q, qd, qdd = perfil_quintico(q0, qf, t, tf)
            pos_ee = np.array([ee_pos(q[:, i]) for i in range(N)])

        elif metodo == "cartesiano":
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

        else:
            # --- CONTROL CINEMÁTICO con jacobiano analítico ---
            #
            #   q̇ = J_A⁻¹(q) (ẋ_d + K·e)        con   e = x_d - x_e
            #
            # x_d(t) = perfil quíntico cartesiano  p0 → pf  (línea recta deseada)
            # ẋ_d(t) = derivada analítica del perfil
            # K       = ganancia diagonal (lazo de error: ė + Ke = 0)
            try:
                K_val = float(self._tr_K.get())
                if K_val <= 0:
                    raise ValueError
            except (ValueError, tk.TclError):
                messagebox.showerror(
                    "K invalida",
                    "La ganancia K del control cinematico debe ser un numero positivo.\n"
                    "Valores tipicos: 5 - 50.")
                return
            K_used = K_val

            # Trayectoria deseada en cartesianas: perfil quintico p0 -> pf
            xd     = np.zeros((N, 3))
            xd_dot = np.zeros((N, 3))
            for k in range(3):
                D = pf[k] - p0[k]
                a3 =  10.0 * D / tf**3
                a4 = -15.0 * D / tf**4
                a5 =   6.0 * D / tf**5
                xd[:,     k] = p0[k] +   a3*t**3 +   a4*t**4 +   a5*t**5
                xd_dot[:,  k] =        3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4

            xd_track = xd  # guardamos para superponer en el plot 3D

            q, qd, qdd, pos_ee, err_ee = control_cinematico(
                q0_deg=q0, xd=xd, xd_dot=xd_dot, t=t,
                K_gain=K_val, dls_lambda=0.01)

            # Calcular qf efectivo alcanzado y error final
            qf_real = q[:, -1]
            err_final = np.linalg.norm(err_ee[-1])
            # Reflejar qf alcanzado (informativo) si estabamos en modo articular
            # de captura (no machaca lo que el usuario tipeo): mejor dejarlo
            src_info = (src_info +
                        f"  |  Control cinematico K={K_val:.2f}  "
                        f"|err_f|={err_final:.3f} mm  "
                        f"qf_real=[{qf_real[0]:.1f},{qf_real[1]:.1f},"
                        f"{qf_real[2]:.1f}]")

        v_ee = np.gradient(pos_ee, t, axis=0)
        speed_ee = np.linalg.norm(v_ee, axis=1)

        grip_events = []
        if self._tr_grip_en.get():
            grip_events = [
                (0.0, self._tr_grip_start.get()),
                (tf,  self._tr_grip_end.get()),
            ]
        self._tr_data = dict(
            t=t, q=q, qd=qd, qdd=qdd,
            q0=q0, qf=qf, tf=tf,
            p0=p0, pf=pf,
            pos_ee=pos_ee, v_ee=v_ee, speed_ee=speed_ee,
            metodo=metodo,
            err_ee=err_ee, K_used=K_used, xd_track=xd_track,
            grip_events=grip_events,
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
        ax.plot(t, d["speed_ee"], color="#d2a8ff", lw=1.5, label="|v| EE")
        # Si es control cinematico, superponer la norma del error cartesiano
        if d.get("err_ee") is not None:
            err_norm = np.linalg.norm(d["err_ee"], axis=1)
            ax2 = ax.twinx()
            ax2.plot(t, err_norm, color="#f85149", lw=1.2, ls="--",
                     label="|e| (mm)")
            ax2.set_ylabel("|e| mm", fontsize=7, color="#f85149")
            ax2.tick_params(axis="y", colors="#f85149", labelsize=6)
            ax2.spines['right'].set_color("#f85149")
            ax.legend(fontsize=6, loc="upper left",
                      facecolor=PANEL, edgecolor=BORDER)
            ax2.legend(fontsize=6, loc="upper right",
                       facecolor=PANEL, edgecolor=BORDER)
            ax.set_title("|v| EE  +  |e| error cartesiano", fontsize=8, pad=3)
        else:
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

        # Si es control cinematico, dibujar tambien la trayectoria deseada x_d(t)
        # (debe coincidir con la linea recta + perfil quintico) y la real seguida
        if d.get("xd_track") is not None:
            xd = d["xd_track"]
            ax3.plot(xd[:, 0], xd[:, 1], xd[:, 2],
                     color="#f0c040", lw=1.0, ls=":", alpha=0.7,
                     label="x_d(t) (deseada)")

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
        if d['metodo'] == "articular":
            titulo = "Interpolacion articular: curva (FK no lineal)"
        elif d['metodo'] == "cartesiano":
            titulo = "Interpolacion cartesiana: linea recta (IK en cada paso)"
        elif d.get("K_used") is not None:
            K = d["K_used"]
            titulo = (f"Control cinematico (Jacobiano)  K={K:.1f}  "
                      f"q\u0307 = J\u207b\u00b9(q)(\u1e8b_d + K\u00b7e)")
        else:
            titulo = str(d['metodo'])
        ax3.set_title(titulo, fontsize=9, color=ACCENT, pad=6)
        ax3.tick_params(labelsize=7)
        for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
            pane.set_facecolor("#0a0f16"); pane.set_edgecolor(BORDER)

        self._canvas_tr.draw_idle()

    def _draw_arm_on(self, ax3, q_deg, first=False):
        """Dibuja/actualiza los eslabones del brazo en la figura 3D de trayectoria."""
        origins = fk(q_deg)          # (5,3): O0,O1,O2,O3,TCP
        xs, ys, zs = origins[:, 0], origins[:, 1], origins[:, 2]

        if first:
            self._tr_arm_lines = []
            # Eslabones articulados O0→O1→O2→O3
            for i in range(3):
                (l,) = ax3.plot([xs[i], xs[i+1]],
                                [ys[i], ys[i+1]],
                                [zs[i], zs[i+1]],
                                color=C_J[i], lw=5, solid_capstyle="round",
                                zorder=4)
                self._tr_arm_lines.append(l)
            # Gripper O3→TCP
            (l_grip,) = ax3.plot([xs[3], xs[4]],
                                 [ys[3], ys[4]],
                                 [zs[3], zs[4]],
                                 color="#ffa657", lw=3, linestyle="--",
                                 solid_capstyle="round", zorder=4,
                                 label=f"gripper ({L_GRIPPER:.0f} mm)")
            self._tr_arm_lines.append(l_grip)
            # Articulaciones O0..O3
            self._tr_arm_joints = ax3.scatter(
                xs[:4], ys[:4], zs[:4], color=ACCENT,
                s=70, edgecolors="#000", linewidth=1, zorder=5)
            # TCP
            self._tr_arm_ee = ax3.scatter(
                [xs[4]], [ys[4]], [zs[4]], color="#ffa657",
                s=140, marker="*", edgecolors="#000", linewidth=1, zorder=6)
        else:
            # Eslabones articulados
            for i in range(3):
                self._tr_arm_lines[i].set_data_3d([xs[i], xs[i+1]],
                                                   [ys[i], ys[i+1]],
                                                   [zs[i], zs[i+1]])
            # Gripper
            self._tr_arm_lines[3].set_data_3d([xs[3], xs[4]],
                                               [ys[3], ys[4]],
                                               [zs[3], zs[4]])
            self._tr_arm_joints._offsets3d = (xs[:4], ys[:4], zs[:4])
            self._tr_arm_ee._offsets3d     = ([xs[4]], [ys[4]], [zs[4]])

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

    # ═══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA 4 — SERIAL / ESP32
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_tab_serial(self, parent):
        # ── Panel superior: conexion ──
        top = tk.Frame(parent, bg=PANEL, pady=8); top.pack(fill="x", side="top")

        tk.Label(top, text="Puerto:", bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(14, 4))
        self._ser_port = ttk.Combobox(top, width=22,
                                      font=("Courier New", 9), state="readonly")
        self._ser_port.pack(side="left", padx=4)
        tk.Button(top, text="↻",
                  bg=BTN_BG, fg=TEXT, relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=8, pady=4,
                  command=self._refresh_ports).pack(side="left", padx=2)

        tk.Label(top, text="  Baud:", bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(10, 4))
        self._ser_baud = tk.StringVar(value="115200")
        tk.Entry(top, textvariable=self._ser_baud, width=8,
                 bg="#0a0e14", fg=ACCENT, insertbackground=ACCENT,
                 relief="flat", font=("Courier New", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=BTN_HOV).pack(side="left", padx=4)

        self._btn_connect = tk.Button(
            top, text="  Conectar  ",
            bg=BTN_PLAY, fg=ACCENT, relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=14, pady=6,
            command=self._toggle_serial)
        self._btn_connect.pack(side="left", padx=10)

        self._ser_status = tk.Label(top, text="desconectado", bg=PANEL, fg="#f85149",
                                    font=("Courier New", 10, "bold"))
        self._ser_status.pack(side="left", padx=8)

        # Boton STOP (gigante)
        tk.Button(top, text="  ■ STOP / RESET  ",
                  bg="#b62324", fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=14, pady=6,
                  activebackground="#da3633", activeforeground=ACCENT,
                  command=self._serial_stop).pack(side="right", padx=14)

        # ── Cuerpo: columna izquierda (control) + derecha (telemetria) ──
        body = tk.Frame(parent, bg=BG); body.pack(fill="both", expand=True)

        # ══ IZQUIERDA: controles de envio ══
        # Panel scrollable: fundamental para que en pantallas chicas
        # (1366x768, 1280x720, 1024x768...) todos los controles sigan
        # accesibles. Sin esto, el bloque MODO JOG mas todo lo demas
        # excede la altura disponible y los controles inferiores
        # (Sincronizar, Zero todos, Streaming) quedan recortados.
        #
        # Truco para minimizar cambios: el ScrollableFrame externo se llama
        # "left_outer", y reasignamos "left" a left_outer.inner para que el
        # resto del codigo (que crea muchos tk.Frame(left, ...)) siga
        # funcionando sin modificaciones.
        left_outer = ScrollableFrame(body, bg=PANEL)
        left_outer.configure(width=540)   # ancho del area visible (incluye scrollbar)
        left_outer.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left_outer.pack_propagate(False)
        left = left_outer.inner

        # Habilitar scroll con la rueda del raton cuando el cursor esta
        # sobre el panel izquierdo. Sin esto el scrollbar funciona pero
        # es engorroso de usar.
        def _scroll_left(evt):
            delta = -1 if (evt.delta < 0 or getattr(evt, "num", 0) == 5) else 1
            try: left_outer._canvas.yview_scroll(-delta, "units")
            except tk.TclError: pass
        # Bind a todos los hijos: cuando el raton entra en cualquiera, capturamos
        # la rueda. Usamos bind_class para que no haga falta enlazar widget por
        # widget.
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            left_outer._canvas.bind(ev, _scroll_left)
            left_outer.inner.bind(ev, _scroll_left)
        # Propagacion: cuando el raton entra en el panel, redirigimos los
        # eventos de scroll al canvas.
        def _bind_scroll_recursive(widget):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try: widget.bind(ev, _scroll_left)
                except tk.TclError: pass
            for ch in widget.winfo_children():
                _bind_scroll_recursive(ch)
        # Lo aplicamos despues de construir el resto del panel; lo dejamos
        # programado con un after para que se ejecute cuando ya este todo creado.
        self.after(200, lambda: _bind_scroll_recursive(left_outer.inner))

        # ── Toggle de modo: Sliders / JOG ─────────────────────────────
        # Modo "sliders": control normal con rangos simetricos +/-rango/2.
        # Modo "jog": control con teclado, util para alcanzar posiciones
        # fuera del rango simetrico del slider o sin saber la cifra exacta.
        self._ctrl_mode = tk.StringVar(value="sliders")
        toggle_frm = tk.Frame(left, bg=PANEL); toggle_frm.pack(fill="x", pady=(8, 0))
        tk.Label(toggle_frm, text="MODO:", bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(side="left", padx=(14, 6))
        for lbl, val in [("Sliders", "sliders"), ("JOG (teclado)", "jog")]:
            tk.Radiobutton(toggle_frm, text=lbl, variable=self._ctrl_mode, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2",
                           command=self._on_ctrl_mode_change
                           ).pack(side="left", padx=4)

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=(10, 0), padx=14)

        tk.Label(left, text="CONTROL INDIVIDUAL POR MOTOR",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(pady=(8, 6))

        # Rangos simetricos centrados en 0 para los sliders del control directo.
        # Util tras hacer "zero" al motor en una pose conocida: el cero del
        # encoder pasa a ser el centro del recorrido, y los limites son ±rango/2.
        # Q_MIN/Q_MAX siguen siendo los rangos fisicos asimetricos del brazo
        # (usados en Vista 3D, planificacion y ping-pong).
        rangos = (Q_MAX - Q_MIN) / 2.0     # mitades de recorrido
        ser_lim = [(-r, r) for r in rangos]

        # Guardamos los Scale para poder deshabilitarlos en modo JOG
        self._ser_motor_scales = []
        self._ser_motor_buttons = []     # botones Enviar/Zero (para deshabilitar)
        self._ser_motor_vars = []
        for j in range(3):
            frm = tk.Frame(left, bg=PANEL); frm.pack(fill="x", padx=14, pady=4)
            tk.Label(frm, text=JNAMES[j], bg=PANEL, fg=C_J[j],
                     font=("Courier New", 11, "bold"),
                     width=4).pack(side="left")

            var = tk.DoubleVar(value=0.0)
            self._ser_motor_vars.append(var)
            lo, hi = ser_lim[j]
            scl = tk.Scale(frm, from_=lo, to=hi, resolution=1.0,
                           orient="horizontal", length=180,
                           bg=PANEL, fg=TEXT, troughcolor=BTN_BG,
                           highlightthickness=0, font=("Courier New", 8),
                           variable=var)
            scl.pack(side="left", padx=6)
            self._ser_motor_scales.append(scl)

            b_send = tk.Button(frm, text="Enviar",
                      bg=BTN_HOV, fg=ACCENT, relief="flat", bd=0,
                      font=("Courier New", 9, "bold"), cursor="hand2",
                      padx=8, pady=3,
                      command=lambda j=j: self._send_single(j+1,
                                                             self._ser_motor_vars[j].get()))
            b_send.pack(side="left", padx=4)
            self._ser_motor_buttons.append(b_send)

            # Boton Zero por motor: pone a 0 la posicion ACTUAL del encoder
            # (no mueve el motor; redefine donde esta el cero).
            b_zero = tk.Button(frm, text="⊙ Zero",
                      bg="#7d4a1f", fg=ACCENT, relief="flat", bd=0,
                      font=("Courier New", 9, "bold"), cursor="hand2",
                      padx=8, pady=3,
                      activebackground="#a86b2a", activeforeground=ACCENT,
                      command=lambda j=j: self._zero_motor(j+1))
            b_zero.pack(side="left", padx=2)
            self._ser_motor_buttons.append(b_zero)

        # ──────────────────────────────────────────────────────────────
        # ── MODO JOG (control con teclado) ────────────────────────────
        # ──────────────────────────────────────────────────────────────
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=10, padx=14)

        # Cabecera del modo JOG con indicador de estado
        hdr = tk.Frame(left, bg=PANEL); hdr.pack(fill="x", padx=14, pady=(2, 4))
        tk.Label(hdr, text="MODO JOG",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(side="left")
        self._lbl_jog_state = tk.Label(
            hdr, text=" ●  inactivo",
            bg=PANEL, fg=MUTED,
            font=("Courier New", 9, "bold"))
        self._lbl_jog_state.pack(side="left", padx=8)

        # Paso por pulsacion + recordatorio de teclas
        frm = tk.Frame(left, bg=PANEL); frm.pack(fill="x", padx=14, pady=(2, 4))
        tk.Label(frm, text="paso (°):", bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left")
        self._jog_step = tk.DoubleVar(value=1.0)
        tk.Entry(frm, textvariable=self._jog_step, width=6,
                 bg="#0a0e14", fg=ACCENT, insertbackground=ACCENT,
                 relief="flat", font=("Courier New", 9),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=BTN_HOV).pack(side="left", padx=4)
        # Botones rapidos para paso
        for v in (0.5, 1.0, 5.0, 10.0):
            tk.Button(frm, text=f"{v:g}",
                      bg=BTN_BG, fg=TEXT, relief="flat", bd=0,
                      font=("Courier New", 8), cursor="hand2",
                      padx=4, pady=1,
                      command=lambda x=v: self._jog_step.set(x)
                      ).pack(side="left", padx=1)

        # Recordatorio de teclas
        keys_frm = tk.Frame(left, bg=PANEL)
        keys_frm.pack(fill="x", padx=14, pady=(2, 4))
        tk.Label(keys_frm,
                 text="teclas:  a/d → ±q1    w/s → ±q2    q/e → ±q3    p → servo",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="w")

        # Cuadro de valores actuales (q1/q2/q3 y x/y/z)
        # Visible siempre, pero se "ilumina" cuando el modo JOG esta activo.
        vals_frm = tk.Frame(left, bg="#0a0e14",
                            highlightbackground=BORDER, highlightthickness=1)
        vals_frm.pack(fill="x", padx=14, pady=(4, 6), ipady=4)

        # Linea articular
        row = tk.Frame(vals_frm, bg="#0a0e14"); row.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(row, text="ang  ",
                 bg="#0a0e14", fg=MUTED,
                 font=("Courier New", 9, "bold")).pack(side="left")
        self._jog_lbl_q = []
        for j in range(3):
            tk.Label(row, text=JNAMES[j]+"=",
                     bg="#0a0e14", fg=C_J[j],
                     font=("Courier New", 9)).pack(side="left", padx=(6, 0))
            lbl = tk.Label(row, text="    0.0°", width=8, anchor="w",
                           bg="#0a0e14", fg=ACCENT,
                           font=("Courier New", 9, "bold"))
            lbl.pack(side="left")
            self._jog_lbl_q.append(lbl)

        # Linea cartesiana
        row = tk.Frame(vals_frm, bg="#0a0e14"); row.pack(fill="x", padx=8, pady=(2, 4))
        tk.Label(row, text="xyz  ",
                 bg="#0a0e14", fg=MUTED,
                 font=("Courier New", 9, "bold")).pack(side="left")
        self._jog_lbl_xyz = []
        for axis in ("x", "y", "z"):
            tk.Label(row, text=axis+"=",
                     bg="#0a0e14", fg="#f0c040",
                     font=("Courier New", 9)).pack(side="left", padx=(6, 0))
            lbl = tk.Label(row, text="   0.0mm", width=10, anchor="w",
                           bg="#0a0e14", fg=ACCENT,
                           font=("Courier New", 9, "bold"))
            lbl.pack(side="left")
            self._jog_lbl_xyz.append(lbl)

        # Estado interno del JOG. _jog_q es la pose articular comandada
        # (no la real; la real viene de telemetria). Se actualiza con
        # cada pulsacion y se manda al firmware como posRef.
        self._jog_q       = [0.0, 0.0, 0.0]    # grados
        self._jog_grip_on = False              # toggle del servo
        self._jog_last_send = [0.0, 0.0, 0.0]  # tiempo ultimo envio (throttle)
        self._jog_last_servo_send = 0.0        # tiempo ultimo toggle servo
        self._jog_active = False
        # ────────────────── fin MODO JOG ─────────────────────────────

        # ── GRIPPER (servo MG90) ──────────────────────────────────
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=10, padx=14)

        tk.Label(left, text="GRIPPER  (servo MG90 — GPIO 13)",
                 bg=PANEL, fg="#f0c040",
                 font=("Courier New", 10, "bold")).pack(pady=(4, 6))

        # Slider de posición + botón Enviar (igual que las juntas)
        frm = tk.Frame(left, bg=PANEL); frm.pack(fill="x", padx=14, pady=4)
        tk.Label(frm, text="grip", bg=PANEL, fg="#f0c040",
                 font=("Courier New", 11, "bold"),
                 width=4).pack(side="left")

        self._ser_grip_var = tk.IntVar(value=GRIP_DEF_DEG)
        scl = tk.Scale(frm, from_=GRIP_MIN_DEG, to=GRIP_MAX_DEG, resolution=1,
                       orient="horizontal", length=200,
                       bg=PANEL, fg=TEXT, troughcolor=BTN_BG,
                       highlightthickness=0, font=("Courier New", 8),
                       variable=self._ser_grip_var)
        scl.pack(side="left", padx=6)

        tk.Button(frm, text="Enviar",
                  bg=BTN_HOV, fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=10, pady=3,
                  command=lambda: self._send_grip(self._ser_grip_var.get())
                  ).pack(side="left", padx=6)

        # Botones rápidos: ABRIR / CERRAR
        frm = tk.Frame(left, bg=PANEL); frm.pack(fill="x", padx=14, pady=(2, 4))
        tk.Label(frm, text="     ",
                 bg=PANEL, fg=PANEL, width=4).pack(side="left")  # alinear con slider

        tk.Button(frm, text=f"  ◯ ABRIR  ({GRIP_OPEN_DEG}°)  ",
                  bg="#1f4a3f", fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=8, pady=3,
                  activebackground="#2d6e3a", activeforeground=ACCENT,
                  command=lambda: self._send_grip_quick("abrir")
                  ).pack(side="left", padx=4)

        tk.Button(frm, text=f"  ⬤ CERRAR ({GRIP_CLOSE_DEG}°)  ",
                  bg="#7d4a1f", fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=8, pady=3,
                  activebackground="#a86b2a", activeforeground=ACCENT,
                  command=lambda: self._send_grip_quick("cerrar")
                  ).pack(side="left", padx=4)

        # Indicador de estado actual del gripper (se actualiza con telemetría)
        frm = tk.Frame(left, bg=PANEL); frm.pack(fill="x", padx=14, pady=(2, 0))
        tk.Label(frm, text="     estado:", bg=PANEL, fg=MUTED,
                 font=("Courier New", 9), width=12, anchor="w").pack(side="left")
        self._lbl_grip_state = tk.Label(
            frm, text=f"abierto ({GRIP_OPEN_DEG}°)", bg=PANEL, fg="#3fb950",
            font=("Courier New", 10, "bold"))
        self._lbl_grip_state.pack(side="left", padx=4)

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=10, padx=14)

        tk.Label(left, text="ENVIAR LOS TRES A LA VEZ",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(pady=(4, 6))

        frm = tk.Frame(left, bg=PANEL); frm.pack(fill="x", padx=14, pady=4)
        tk.Button(frm, text="  Sincronizar valores de sliders  ",
                  bg=BTN_HOV, fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=10, pady=5,
                  command=self._send_multi_sliders
                  ).pack(fill="x")

        # Boton "Zero todos" — pone a 0 la posicion ACTUAL de los 3 motores
        # de una sola vez. Util tras posar el brazo en una pose conocida
        # (p.ej. contra finales de carrera).
        frm = tk.Frame(left, bg=PANEL); frm.pack(fill="x", padx=14, pady=(2, 4))
        tk.Button(frm, text="  ⊙ Zero todos los motores (pos actual = 0)  ",
                  bg="#7d4a1f", fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=10, pady=5,
                  activebackground="#a86b2a", activeforeground=ACCENT,
                  command=self._zero_all_motors
                  ).pack(fill="x")

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=10, padx=14)

        tk.Label(left, text="STREAMING DE TRAYECTORIA",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(pady=(4, 4))
        tk.Label(left,
                 text="(Planificala primero en la pestana 3)",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(pady=(0, 6))

        frm_s = tk.Frame(left, bg=PANEL); frm_s.pack(fill="x", padx=14, pady=2)
        tk.Label(frm_s, text="Frecuencia (Hz):", bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(0, 4))
        self._stream_hz = tk.StringVar(value="20")
        tk.Entry(frm_s, textvariable=self._stream_hz, width=5,
                 bg="#0a0e14", fg=ACCENT, insertbackground=ACCENT,
                 relief="flat", font=("Courier New", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=BTN_HOV).pack(side="left")
        tk.Label(frm_s, text=" (10-50 recomendado)", bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(side="left", padx=4)

        frm_s2 = tk.Frame(left, bg=PANEL); frm_s2.pack(fill="x", padx=14, pady=6)
        self._btn_stream = tk.Button(
            frm_s2, text="  ▶ Enviar trayectoria planificada  ",
            bg=BTN_PLAY, fg=ACCENT, relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=10, pady=6,
            command=self._start_stream)
        self._btn_stream.pack(fill="x")

        self._lbl_stream = tk.Label(left, text="", bg=PANEL, fg=MUTED,
                                     font=("Courier New", 9))
        self._lbl_stream.pack(pady=4)

        # ══ DERECHA: telemetria + consola ══
        right = tk.Frame(body, bg=BG); right.pack(side="left", fill="both", expand=True,
                                                   padx=(5, 10), pady=10)

        # Figura de telemetria
        self._fig_ser = plt.Figure(figsize=(7, 3.5), facecolor=BG)
        self._fig_ser.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12,
                                       hspace=0.55)
        self._ax_tel = [self._fig_ser.add_subplot(3, 1, j+1) for j in range(3)]
        for j in range(3):
            self._ax_tel[j].set_title(f"Telemetria {JNAMES[j]} — pos (linea) vs ref (punteada)",
                                       fontsize=8, pad=3)
            self._ax_tel[j].set_ylabel("deg", fontsize=7)
            self._ax_tel[j].tick_params(labelsize=7)
        self._ax_tel[2].set_xlabel("t (s)", fontsize=7)

        self._line_pos = []
        self._line_ref = []
        for j in range(3):
            lp, = self._ax_tel[j].plot([], [], color=C_J[j], lw=1.5, label="pos")
            lr, = self._ax_tel[j].plot([], [], color=MUTED, lw=1.0, ls="--", label="ref")
            self._line_pos.append(lp); self._line_ref.append(lr)
            self._ax_tel[j].legend(fontsize=6.5, facecolor=PANEL, edgecolor=BORDER,
                                    loc="upper left")

        self._canvas_ser = FigureCanvasTkAgg(self._fig_ser, master=right)
        self._canvas_ser.get_tk_widget().pack(fill="both", expand=True)

        # Consola
        cons_frame = tk.Frame(right, bg=PANEL, pady=4)
        cons_frame.pack(fill="x", pady=(6, 0))
        tk.Label(cons_frame, text=" Consola serial (TX azul / RX gris)  ",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 9, "bold")).pack(side="left")
        tk.Button(cons_frame, text="Limpiar",
                  bg=BTN_BG, fg=TEXT, relief="flat", bd=0,
                  font=("Courier New", 8), cursor="hand2",
                  padx=8, pady=2,
                  command=self._clear_console).pack(side="right", padx=6)

        self._console = tk.Text(right, height=9, bg="#0a0e14", fg=TEXT,
                                 insertbackground=ACCENT, relief="flat",
                                 font=("Courier New", 9), bd=0, wrap="none",
                                 highlightthickness=1, highlightbackground=BORDER)
        self._console.pack(fill="x", pady=(2, 0))
        self._console.tag_config("tx", foreground="#58a6ff")
        self._console.tag_config("rx", foreground="#8b949e")
        self._console.tag_config("info", foreground="#3fb950")
        self._console.tag_config("err", foreground="#f85149")

        # Refrescar lista de puertos al inicio
        self._refresh_ports()
        if not SERIAL_OK:
            self._log("info",
                      "  pyserial no instalado. Instalalo con: pip install pyserial")

    # ──────────────────────────────────────────
    #  Helpers de consola y log
    # ──────────────────────────────────────────
    def _log(self, tag, text):
        try:
            self._console.insert("end", text + "\n", tag)
            self._console.see("end")
            # limitar a 400 lineas
            lines = int(self._console.index("end-1c").split(".")[0])
            if lines > 400:
                self._console.delete("1.0", f"{lines-400}.0")
        except tk.TclError:
            pass

    def _clear_console(self):
        try:
            self._console.delete("1.0", "end")
        except tk.TclError:
            pass

    # ──────────────────────────────────────────
    #  Gestión del puerto serial
    # ──────────────────────────────────────────
    def _refresh_ports(self):
        if not SERIAL_OK:
            self._ser_port["values"] = []
            self._ser_port.set("pyserial no instalado")
            return
        ports = list(serial.tools.list_ports.comports())
        names = [f"{p.device}  ({p.description[:30]})" for p in ports]
        self._ser_port["values"] = names
        if names and not self._ser_port.get():
            self._ser_port.current(0)

    def _toggle_serial(self):
        if self._serial is not None and self._serial.is_open:
            self._disconnect_serial()
        else:
            self._connect_serial()

    def _connect_serial(self):
        if not SERIAL_OK:
            messagebox.showerror("pyserial no instalado",
                                 "Para usar esta pestaña instala pyserial:\n\n"
                                 "  pip install pyserial")
            return
        raw = self._ser_port.get()
        if not raw:
            messagebox.showerror("Sin puerto", "Seleccioná un puerto primero.")
            return
        portname = raw.split()[0]    # primer token = device
        try:
            baud = int(self._ser_baud.get())
        except ValueError:
            messagebox.showerror("Baud invalido", "Baud rate debe ser un numero.")
            return
        try:
            self._serial = serial.Serial(portname, baud, timeout=0.1)
        except Exception as e:
            messagebox.showerror("Error al conectar", str(e))
            self._serial = None
            return

        self._ser_status.config(text=f"conectado: {portname} @ {baud}", fg="#3fb950")
        self._btn_connect.config(text="  Desconectar  ", bg=BTN_PAUS)
        self._log("info", f"[info] Conectado a {portname} @ {baud}")

        # Arrancar hilo de lectura
        self._serial_rx_stop.clear()
        self._serial_rx_thread = threading.Thread(
            target=self._rx_worker, daemon=True)
        self._serial_rx_thread.start()

        # Arrancar drenado de la cola desde la UI
        self._tel_t0 = time.monotonic()
        self._tel_idx = 0
        self._schedule_pump()

    def _disconnect_serial(self):
        # Parar stream si esta corriendo
        self._stream_stop.set()

        # Parar hilo RX
        self._serial_rx_stop.set()
        if self._serial_rx_thread is not None:
            self._serial_rx_thread.join(timeout=0.5)
            self._serial_rx_thread = None

        # Cancelar pump
        if self._serial_pump_job is not None:
            try: self.after_cancel(self._serial_pump_job)
            except Exception: pass
            self._serial_pump_job = None

        # Cerrar puerto
        if self._serial is not None:
            try: self._serial.close()
            except Exception: pass
            self._serial = None

        self._ser_status.config(text="desconectado", fg="#f85149")
        self._btn_connect.config(text="  Conectar  ", bg=BTN_PLAY)
        self._log("info", "[info] Desconectado")

    def _rx_worker(self):
        """Hilo que lee lineas del serial y las mete en la cola."""
        buf = b""
        while not self._serial_rx_stop.is_set():
            try:
                chunk = self._serial.read(128)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    text = line.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
                if text:
                    self._serial_rx_queue.put(text)

    def _schedule_pump(self):
        if self._serial is None or not self._serial.is_open:
            return
        self._pump_serial_queue()
        self._serial_pump_job = self.after(30, self._schedule_pump)

    def _pump_serial_queue(self):
        """Drena la cola de RX y actualiza la UI (se corre en main thread)."""
        updated = False
        while True:
            try:
                line = self._serial_rx_queue.get_nowait()
            except queue.Empty:
                break

            self._log("rx", "  RX ← " + line)

            # Intentar parsear como JSON de telemetria
            try:
                d = json.loads(line)
                # Formato: {"m1":{"posRef":..,"pos":..,"ok":..}, "m2":{...}, "m3":{...}}
                if isinstance(d, dict) and "m1" in d and "m2" in d and "m3" in d:
                    now = time.monotonic() - self._tel_t0
                    for j, key in enumerate(["m1", "m2", "m3"]):
                        sub = d[key]
                        # El firmware manda pulsos; convertimos a grados
                        pos_deg = float(sub.get("pos", 0))    / PPR[j] * 360.0
                        ref_deg = float(sub.get("posRef", 0)) / PPR[j] * 360.0
                        self._tel[key]["pos"] = pos_deg
                        self._tel[key]["ref"] = ref_deg
                        self._tel[key]["ok"]  = sub.get("ok", True)
                        # Guardar en buffer circular
                        idx = self._tel_idx % self._tel_max
                        self._tel_t[idx]   = now
                        self._tel_pos[j, idx] = pos_deg
                        self._tel_ref[j, idx] = ref_deg
                    self._tel_idx += 1
                    updated = True

                    # Estado del gripper (si el firmware lo envia)
                    if isinstance(d.get("grip"), dict):
                        gpos = int(d["grip"].get("pos", self._tel["grip"]["pos"]))
                        gabi = bool(d["grip"].get("abierto",
                                                   gpos <= GRIP_MIN_DEG + 5))
                        self._tel["grip"]["pos"]     = gpos
                        self._tel["grip"]["abierto"] = gabi
                        self._update_grip_label(gpos, abierto=gabi)
            except (ValueError, TypeError):
                pass  # no era JSON de telemetria, solo mostrar en consola

        if updated:
            self._refresh_telemetry_plot()
            # Refrescar la casilla del modo JOG con los valores reales
            if getattr(self, "_jog_active", False):
                try: self._jog_refresh_labels()
                except Exception: pass

    def _refresh_telemetry_plot(self):
        """Redibuja las graficas de telemetria con el buffer actual."""
        n = min(self._tel_idx, self._tel_max)
        if n < 2:
            return
        # Reordenar buffer circular
        if self._tel_idx <= self._tel_max:
            t   = self._tel_t[:n]
            pos = self._tel_pos[:, :n]
            ref = self._tel_ref[:, :n]
        else:
            start = self._tel_idx % self._tel_max
            t   = np.concatenate([self._tel_t[start:],   self._tel_t[:start]])
            pos = np.concatenate([self._tel_pos[:, start:], self._tel_pos[:, :start]], axis=1)
            ref = np.concatenate([self._tel_ref[:, start:], self._tel_ref[:, :start]], axis=1)

        for j in range(3):
            self._line_pos[j].set_data(t, pos[j])
            self._line_ref[j].set_data(t, ref[j])
            self._ax_tel[j].set_xlim(t[0], max(t[-1], t[0]+0.1))
            # y autoscale con margen
            combined = np.concatenate([pos[j], ref[j]])
            lo, hi = combined.min(), combined.max()
            span = max(10.0, hi - lo)
            c = (lo + hi) / 2
            self._ax_tel[j].set_ylim(c - span*0.6, c + span*0.6)
        self._canvas_ser.draw_idle()

    # ──────────────────────────────────────────
    #  Envio de comandos
    # ──────────────────────────────────────────
    def _send_json(self, obj):
        if self._serial is None or not self._serial.is_open:
            messagebox.showwarning("Sin conexion",
                                   "Conectate al ESP32 primero.")
            return False
        try:
            line = json.dumps(obj, separators=(",", ":")) + "\n"
            self._serial.write(line.encode("utf-8"))
            self._log("tx", "TX → " + line.rstrip())
            return True
        except Exception as e:
            self._log("err", f"[err] TX fallo: {e}")
            return False

    def _send_single(self, motor_id, pos_deg):
        self._send_json({"motor": motor_id, "pos": round(float(pos_deg), 2)})

    def _send_multi_sliders(self):
        vals = [round(float(v.get()), 2) for v in self._ser_motor_vars]
        self._send_json({"m1": vals[0], "m2": vals[1], "m3": vals[2]})

    # ──────────────────────────────────────────────────────────────
    #  MODO JOG  (control con teclado)
    # ──────────────────────────────────────────────────────────────
    #
    #  Idea: en lugar de mover sliders, el usuario pulsa teclas
    #  (a/d, w/s, q/e) y cada pulsacion aplica un "paso" angular
    #  configurable. Util para llegar a posiciones que estan fuera
    #  del rango simetrico de los sliders, o para hacer ajustes
    #  finos sin escribir cifras exactas.
    #
    #  El modo se activa con un toggle. Mientras esta activo:
    #    - Las teclas se atan a la ventana raiz con bind_all.
    #    - Los sliders (y sus botones Enviar/Zero) se deshabilitan
    #      visualmente para evitar comandos contradictorios.
    #    - Una casilla muestra los valores articulares y XYZ
    #      reales del brazo (calculados a partir de la telemetria).
    #
    #  Mapeo de teclas:
    #      a / d  ->  -q1 / +q1
    #      w / s  ->  +q2 / -q2
    #      q / e  ->  +q3 / -q3
    #      p      ->  toggle gripper (abrir <-> cerrar)
    # ──────────────────────────────────────────────────────────────

    # Mapeo: keysym -> (motor_idx 0..2, signo)
    _JOG_KEYMAP = {
        "a": (0, -1),  "d": (0, +1),     # q1
        "w": (1, +1),  "s": (1, -1),     # q2
        "q": (2, +1),  "e": (2, -1),     # q3
    }
    # Throttle: tiempo minimo (s) entre envios al mismo motor.
    # 30 ms = 33 mensajes/s por motor; el firmware procesa muy holgado.
    _JOG_MIN_INTERVAL = 0.030

    def _on_ctrl_mode_change(self):
        """Llamado cuando el usuario cambia el toggle Sliders <-> JOG."""
        if self._ctrl_mode.get() == "jog":
            self._jog_enable()
        else:
            self._jog_disable()

    def _jog_enable(self):
        """Activa el modo JOG: liga teclas, deshabilita sliders, sincroniza
        la pose comandada con la telemetria actual."""
        if self._jog_active:
            return
        self._jog_active = True

        # Inicializar la pose comandada del JOG con los angulos REALES
        # actuales del brazo (de la telemetria). Asi el primer movimiento
        # del JOG es relativo a donde esta el brazo, no al ultimo destino
        # de los sliders.
        for j in range(3):
            self._jog_q[j] = float(self._tel.get(f"m{j+1}", {}).get("pos", 0.0))

        # Estado del gripper: heredar de la telemetria
        self._jog_grip_on = bool(self._tel.get("grip", {}).get("pos", 0)
                                 > (GRIP_OPEN_DEG + GRIP_CLOSE_DEG) / 2)

        # Deshabilitar sliders y botones del modo Sliders
        for s in self._ser_motor_scales:
            try: s.config(state="disabled", troughcolor="#0a0e14")
            except tk.TclError: pass
        for b in self._ser_motor_buttons:
            try: b.config(state="disabled")
            except tk.TclError: pass

        # Bind teclas a nivel global. bind_all asegura que funcionan
        # aunque el foco este en otro widget.
        for k in list(self._JOG_KEYMAP.keys()) + ["p", "P"]:
            self.bind_all(f"<KeyPress-{k}>", self._jog_on_key)
        # Tambien las versiones mayusculas por si hay Shift activado
        for k in list(self._JOG_KEYMAP.keys()):
            self.bind_all(f"<KeyPress-{k.upper()}>", self._jog_on_key)

        self.focus_set()
        self._lbl_jog_state.config(text=" ●  ACTIVO", fg="#3fb950")
        self._log("info", "[info] modo JOG: activado. teclas a/d w/s q/e p")
        # Refrescar la casilla de valores
        self._jog_refresh_labels()

    def _jog_disable(self):
        """Desactiva el modo JOG y vuelve al modo sliders."""
        if not self._jog_active:
            return
        self._jog_active = False

        # Liberar teclas
        for k in list(self._JOG_KEYMAP.keys()) + ["p", "P"]:
            try: self.unbind_all(f"<KeyPress-{k}>")
            except tk.TclError: pass
            try: self.unbind_all(f"<KeyPress-{k.upper()}>")
            except tk.TclError: pass

        # Re-habilitar sliders
        for s in self._ser_motor_scales:
            try: s.config(state="normal", troughcolor=BTN_BG)
            except tk.TclError: pass
        for b in self._ser_motor_buttons:
            try: b.config(state="normal")
            except tk.TclError: pass

        self._lbl_jog_state.config(text=" ●  inactivo", fg=MUTED)
        self._log("info", "[info] modo JOG: desactivado")

    def _jog_on_key(self, event):
        """Maneja una pulsacion de tecla en modo JOG."""
        if not self._jog_active:
            return
        key = event.keysym.lower()
        now = time.time()

        # Tecla "p" -> toggle del gripper (con debounce mas estricto:
        # 250 ms para evitar que el autorepeat de la tecla mantenida
        # alterne 30 veces por segundo).
        if key == "p":
            if now - self._jog_last_servo_send < 0.25:
                return
            self._jog_last_servo_send = now
            self._jog_grip_on = not self._jog_grip_on
            cmd = "cerrar" if self._jog_grip_on else "abrir"
            self._send_grip_quick(cmd)
            return

        # Movimiento articular
        if key in self._JOG_KEYMAP:
            j, signo = self._JOG_KEYMAP[key]

            # Throttle por motor para no saturar el serial
            if now - self._jog_last_send[j] < self._JOG_MIN_INTERVAL:
                return
            self._jog_last_send[j] = now

            try:
                paso = abs(float(self._jog_step.get()))
            except (ValueError, tk.TclError):
                paso = 1.0
            paso = max(0.1, min(20.0, paso))

            # Actualizar pose comandada y enviar al firmware
            self._jog_q[j] += signo * paso
            self._send_single(j + 1, self._jog_q[j])

            # Refrescar la casilla de valores con la pose COMANDADA
            # (la real llegara con la telemetria)
            self._jog_refresh_labels()

    def _jog_refresh_labels(self):
        """Actualiza la casilla q1/q2/q3 + x/y/z. Usa los valores REALES
        del brazo (de telemetria) cuando esta disponible; si no, usa la
        pose comandada como aproximacion."""
        q_real = []
        for j in range(3):
            v = self._tel.get(f"m{j+1}", {}).get("pos", None)
            if v is None:
                v = self._jog_q[j]
            q_real.append(float(v))

        # Actualizar etiquetas articulares
        for j in range(3):
            try:
                self._jog_lbl_q[j].config(text=f"{q_real[j]:+7.1f}°")
            except tk.TclError: pass

        # Calcular XYZ con cinematica directa
        try:
            xyz = ee_pos(np.array(q_real))
            for k in range(3):
                self._jog_lbl_xyz[k].config(text=f"{xyz[k]:+8.1f}mm")
        except Exception:
            for k in range(3):
                try: self._jog_lbl_xyz[k].config(text="     ---")
                except tk.TclError: pass

    def _zero_motor(self, motor_id):
        """Pone a 0 la posicion ACTUAL del encoder del motor indicado.
        No mueve el motor: redefine donde esta el cero. Usa el comando
        {"zero": N} que entiende el firmware. Tambien deja el slider de
        ese motor a 0 para que coincida con la nueva referencia."""
        if motor_id not in (1, 2, 3):
            return
        self._send_json({"zero": motor_id})
        # Sincronizar el slider local: la pos del motor pasa a ser 0,
        # asi que tambien el control directo apunta a 0.
        try:
            self._ser_motor_vars[motor_id - 1].set(0.0)
        except (IndexError, tk.TclError):
            pass
        self._log("info", f"[info] zero M{motor_id}: pos actual fijada como origen")

    def _zero_all_motors(self):
        """Pone a 0 la posicion actual de los 3 motores de una sola vez."""
        if not messagebox.askyesno(
                "Confirmar zero",
                "¿Fijar la posición ACTUAL de los 3 motores como nuevo cero?\n\n"
                "Los motores no se moverán. El encoder pasará a contar desde 0\n"
                "en la pose actual del brazo.\n\n"
                "Útil tras llevar el brazo manualmente a una pose de referencia\n"
                "conocida (p.ej. contra finales de carrera)."):
            return
        self._send_json({"zero": "all"})
        # Reset de los sliders locales
        for v in self._ser_motor_vars:
            try: v.set(0.0)
            except tk.TclError: pass
        self._log("info", "[info] zero all: pos actual de M1/M2/M3 fijadas como origen")

    def _send_grip(self, deg):
        """Envia un angulo concreto al servo del gripper [0..180]."""
        try:
            d = int(round(float(deg)))
        except (ValueError, TypeError):
            self._log("info", "[info] gripper: valor invalido")
            return
        d = max(GRIP_MIN_DEG, min(GRIP_MAX_DEG, d))
        self._send_json({"grip": d})
        # Estado optimista: refrescar la etiqueta sin esperar a la telemetria.
        # Cuando llegue el siguiente paquete con "grip" se reescribira.
        self._update_grip_label(d, abierto=(d <= GRIP_OPEN_DEG + 5))

    def _send_grip_quick(self, accion):
        """Envia los comandos string 'abrir' o 'cerrar' (los entiende el firmware)."""
        if accion not in ("abrir", "cerrar"):
            return
        self._send_json({"grip": accion})
        # Sincronizar el slider con el comando rápido y actualizar la etiqueta
        target = GRIP_OPEN_DEG if accion == "abrir" else GRIP_CLOSE_DEG
        self._ser_grip_var.set(target)
        self._update_grip_label(target, abierto=(accion == "abrir"))

    def _update_grip_label(self, deg, abierto=None):
        """Actualiza el indicador de estado del gripper (texto + color)."""
        if not hasattr(self, "_lbl_grip_state"):
            return
        if abierto is None:
            abierto = (deg <= GRIP_OPEN_DEG + 5)
        cerrado_total = (deg >= GRIP_CLOSE_DEG - 5)
        if abierto:
            txt, col = f"abierto ({deg}°)", "#3fb950"
        elif cerrado_total:
            txt, col = f"cerrado ({deg}°)", "#f0c040"
        else:
            txt, col = f"intermedio ({deg}°)", "#d2a8ff"
        try:
            self._lbl_grip_state.config(text=txt, fg=col)
        except tk.TclError:
            pass

    def _serial_stop(self):
        """STOP: corta streaming, pone las referencias en la pos actual (frena)."""
        self._stream_stop.set()
        if self._serial is not None and self._serial.is_open:
            # Mandar como nueva referencia la posicion actual leida (para que el
            # controlador se detenga donde este). Si no hay telemetria, reset a 0.
            if self._tel_idx > 0:
                cmd = {"m1": round(self._tel["m1"]["pos"], 2),
                       "m2": round(self._tel["m2"]["pos"], 2),
                       "m3": round(self._tel["m3"]["pos"], 2)}
                self._send_json(cmd)
                self._log("info", "[info] STOP: referencias fijadas a pos actual")
            else:
                self._send_json({"cmd": "reset"})
                self._log("info", "[info] STOP: comando reset enviado")

    # ──────────────────────────────────────────
    #  Streaming de trayectoria (hilo separado)
    # ──────────────────────────────────────────
    def _start_stream(self):
        has_serial = self._serial is not None and self._serial.is_open
        if not has_serial and not self._ros2_running:
            messagebox.showwarning("Sin conexion",
                                   "Conecta el ESP32 o ROS2 primero.")
            return
        if self._tr_data is None:
            messagebox.showwarning("Sin trayectoria",
                                   "Primero planifica una trayectoria en la pestaña 3.")
            return
        try:
            hz = float(self._stream_hz.get())
            if hz <= 0 or hz > 200:
                raise ValueError
        except ValueError:
            messagebox.showerror("Frecuencia invalida", "Usá un valor entre 1 y 200 Hz.")
            return

        # No permitir si ya hay stream
        if self._stream_thread is not None and self._stream_thread.is_alive():
            messagebox.showwarning("Stream en curso",
                                   "Ya hay una trayectoria en ejecucion.")
            return

        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_worker, args=(hz,), daemon=True)
        self._stream_thread.start()
        self._log("info", f"[info] Stream iniciado a {hz} Hz")

    def _stream_worker(self, hz):
        """Hilo: envia setpoints a frecuencia constante respetando el perfil."""
        d = self._tr_data
        if d is None:
            return
        q = d["q"]                  # (3, N) en grados
        tt = d["t"]                 # (N,) tiempos del perfil
        tf = d["tf"]
        N = len(tt)

        period = 1.0 / hz
        t_start = time.monotonic()
        sent = 0
        # Unified grip events: list of (time_s, "abrir"/"cerrar") sorted by time
        grip_events = sorted(d.get("grip_events") or [], key=lambda e: e[0])
        g_idx = 0

        try:
            # Fire any events scheduled at t=0
            while g_idx < len(grip_events) and grip_events[g_idx][0] <= 0.0:
                self._send_json_threadsafe({"grip": grip_events[g_idx][1]})
                g_idx += 1

            while not self._stream_stop.is_set():
                elapsed = time.monotonic() - t_start
                # Fire grip events whose time has been reached
                while g_idx < len(grip_events) and elapsed >= grip_events[g_idx][0]:
                    self._send_json_threadsafe({"grip": grip_events[g_idx][1]})
                    g_idx += 1
                if elapsed >= tf:
                    # Enviar el ultimo punto explicito y terminar
                    cmd = {"m1": round(float(q[0, -1]), 2),
                           "m2": round(float(q[1, -1]), 2),
                           "m3": round(float(q[2, -1]), 2)}
                    self._send_json_threadsafe(cmd)
                    # Fire any remaining grip events (e.g. final close/open)
                    while g_idx < len(grip_events):
                        self._send_json_threadsafe({"grip": grip_events[g_idx][1]})
                        g_idx += 1
                    break
                # Interpolar el frame correspondiente al instante elapsed
                # (tt es uniforme con paso DT)
                i = min(int(elapsed / (tt[1] - tt[0])), N - 1)
                cmd = {"m1": round(float(q[0, i]), 2),
                       "m2": round(float(q[1, i]), 2),
                       "m3": round(float(q[2, i]), 2)}
                self._send_json_threadsafe(cmd)
                if self._ros2_running:
                    self._ros2_publish_q(q[:, i])
                sent += 1

                # Feedback en UI
                try:
                    self._lbl_stream.after(0, lambda s=sent, el=elapsed, N=N:
                                            self._lbl_stream.config(
                                                text=f"enviados: {s}  |  t={el:.2f}s / {tf:.1f}s"))
                except Exception:
                    pass

                # Dormir hasta el proximo tick
                next_tick = t_start + (sent) * period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    if self._stream_stop.wait(sleep_s):
                        break
        except Exception as e:
            try: self.after(0, lambda: self._log("err", f"[err] stream: {e}"))
            except Exception: pass

        try:
            self.after(0, lambda: self._log(
                "info", f"[info] Stream finalizado  (setpoints enviados: {sent})"))
        except Exception:
            pass

    def _send_json_threadsafe(self, obj):
        """Envio de JSON desde un hilo sin actualizar la UI directamente."""
        if self._serial is None or not self._serial.is_open:
            return False
        try:
            line = json.dumps(obj, separators=(",", ":")) + "\n"
            self._serial.write(line.encode("utf-8"))
            # Loguear en la UI de forma thread-safe (via after)
            try:
                self.after(0, lambda s=line.rstrip(): self._log("tx", "TX → " + s))
            except Exception:
                pass
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA 5 — TRAYECTORIA PING PONG (WAYPOINTS PRE-CARGADOS)
    #
    #  Secuencia completa de 3 ping pongs:
    #    REPOSO → sube → [recoger p_i → depositar P_i → subir]×3 → REPOSO
    #    Total: 22 waypoints, 21 tramos, 31.5s a 1.5s/tramo
    #    Z = 40 mm para recogida/depósito, Z = 80 mm para tránsito
    # ═══════════════════════════════════════════════════════════════════════════

    # Waypoints cartesianos (mm) — definidos como constantes de clase para
    # tenerlos siempre disponibles. Coordenadas idénticas al spec del usuario.
    _PP_REPOSO      = (  98.0,    0.0, 157.0)   # pose inicial q=[0°, 117°, -118°]

    # Posiciones de recogida (minúscula) — Z = 40 mm
    _PP_p1          = ( 221.0,   16.0,  40.0)
    _PP_p2          = ( 221.0,    0.0,  40.0)
    _PP_p3          = ( 221.0,  -16.0,  40.0)

    # Posiciones en bandeja (MAYÚSCULA) — Z = 40 mm
    _PP_P1          = ( -23.0,  219.0,  40.0)
    _PP_P2          = ( -42.0,  186.0,  40.0)
    _PP_P3          = (  15.0,  186.0,  40.0)

    # Alturas de tránsito (misma XY, Z = 80 mm)
    _PP_p1_up       = ( 221.0,   16.0,  80.0)
    _PP_p2_up       = ( 221.0,    0.0,  80.0)
    _PP_p3_up       = ( 221.0,  -16.0,  80.0)
    _PP_P1_up       = ( -23.0,  219.0,  80.0)
    _PP_P2_up       = ( -42.0,  186.0,  80.0)
    _PP_P3_up       = (  15.0,  186.0,  80.0)
    _PP_REPOSO_up   = (  98.0,    0.0,  80.0)

    def _pp_default_waypoints(self):
        """Lista de los 22 waypoints predeterminados del ping pong.
        Formato: [nombre, [v1,v2,v3], descripcion, tipo]
        tipo = "xyz"  → v1,v2,v3 son X,Y,Z en mm  (se resuelve IK internamente)
        tipo = "ang"  → v1,v2,v3 son q1,q2,q3 en grados  (se usa directo como keyframe)
        """
        return [
            ["REPOSO",      list(self._PP_REPOSO),      "inicio",                    "xyz"],
            ["REPOSO_up",   list(self._PP_REPOSO_up),   "subir",                     "xyz"],

            ["p1_up",       list(self._PP_p1_up),       "ir sobre p1",               "xyz"],
            ["p1",          list(self._PP_p1),          "bajar y recoger p1",         "xyz"],
            ["p1_up",       list(self._PP_p1_up),       "subir con p1",               "xyz"],

            ["P1_up",       list(self._PP_P1_up),       "ir sobre bandeja slot 1",   "xyz"],
            ["P1",          list(self._PP_P1),          "bajar y soltar en P1",       "xyz"],
            ["P1_up",       list(self._PP_P1_up),       "subir",                     "xyz"],

            ["p2_up",       list(self._PP_p2_up),       "ir sobre p2",               "xyz"],
            ["p2",          list(self._PP_p2),          "bajar y recoger p2",         "xyz"],
            ["p2_up",       list(self._PP_p2_up),       "subir con p2",               "xyz"],

            ["P2_up",       list(self._PP_P2_up),       "ir sobre bandeja slot 2",   "xyz"],
            ["P2",          list(self._PP_P2),          "bajar y soltar en P2",       "xyz"],
            ["P2_up",       list(self._PP_P2_up),       "subir",                     "xyz"],

            ["p3_up",       list(self._PP_p3_up),       "ir sobre p3",               "xyz"],
            ["p3",          list(self._PP_p3),          "bajar y recoger p3",         "xyz"],
            ["p3_up",       list(self._PP_p3_up),       "subir con p3",               "xyz"],

            ["P3_up",       list(self._PP_P3_up),       "ir sobre bandeja slot 3",   "xyz"],
            ["P3",          list(self._PP_P3),          "bajar y soltar en P3",       "xyz"],
            ["P3_up",       list(self._PP_P3_up),       "subir",                     "xyz"],

            ["REPOSO_up",   list(self._PP_REPOSO_up),   "regresar sobre reposo",     "xyz"],
            ["REPOSO",      list(self._PP_REPOSO),      "bajar a reposo final",       "xyz"],
        ]

    def _pp_waypoint_list(self):
        """Devuelve la secuencia actual de waypoints como (nombre, (x,y,z), descripcion).
        Si el waypoint es de tipo 'ang', convierte q1,q2,q3 → XYZ via cinemática directa.
        Si aun no se han inicializado, usa los 22 default. La lista interna es
        editable: self._pp_waypoints (cada item es [nombre, [v1,v2,v3], desc, tipo])."""
        if not hasattr(self, "_pp_waypoints") or self._pp_waypoints is None:
            self._pp_waypoints = self._pp_default_waypoints()
        result = []
        for i, wp in enumerate(self._pp_waypoints):
            name = wp[0]
            vals = wp[1]
            desc = wp[2]
            tipo = wp[3] if len(wp) > 3 else "xyz"
            if tipo == "ang":
                # Convertir ángulos articulares → posición cartesiana via FK
                xyz = tuple(ee_pos(np.array(vals, dtype=float)))
            else:
                xyz = tuple(vals)
            result.append((name, xyz, f"#{i:3d}  {desc}"))
        return result

    def _build_tab_pingpong(self, parent):
        # ── Panel superior: título + descripción ──
        top = tk.Frame(parent, bg=PANEL)
        top.pack(fill="x", side="top")

        tk.Frame(top, bg=BORDER, height=1).pack(fill="x")
        hdr = tk.Frame(top, bg=PANEL, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="TRAYECTORIA PING PONG — 3 ciclos completos",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 13, "bold")).pack(side="left", padx=14)
        tk.Label(hdr,
                 text=" | recoger p_i → depositar P_i → reposo  "
                      " | Z=40mm recogida/dep.  Z=80mm tránsito",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left")
        tk.Frame(top, bg=BORDER, height=1).pack(fill="x")

        # ── Panel de control (botón grande + parámetros, envuelve a varias filas) ──
        ctrl_outer = tk.Frame(parent, bg="#0d1a0d", pady=10)
        ctrl_outer.pack(fill="x", side="top")
        ctrl = WrappingFrame(ctrl_outer, bg="#0d1a0d", hgap=8, vgap=4)
        ctrl.pack(fill="x", padx=10)

        # Botón principal — UN SOLO CLIC
        self._btn_pp_csv = tk.Button(
            ctrl, text="  💾  GENERAR CSV (1 clic)  ",
            bg=BTN_PLAY, fg=ACCENT,
            relief="flat", bd=0,
            font=("Courier New", 12, "bold"), cursor="hand2",
            padx=18, pady=10,
            command=self._pingpong_generate_csv)
        ctrl.add(self._btn_pp_csv)

        # Parámetros editables (seg/tramo y codo arriba/abajo)
        frm_dur, self._pp_dur_var = make_labeled_entry(ctrl, "seg/tramo", 1.5, width=5)
        frm_dur.configure(bg="#0d1a0d")
        for child in frm_dur.winfo_children():
            try: child.configure(bg="#0d1a0d")
            except tk.TclError: pass
        ctrl.add(frm_dur)

        ctrl.add(tk.Frame(ctrl, bg=BORDER, width=2, height=30))

        ctrl.add(tk.Label(ctrl, text="Codo:", bg="#0d1a0d", fg=MUTED,
                          font=("Courier New", 9)))
        self._pp_elbow = tk.StringVar(value="arriba")
        for modo in ("arriba", "abajo"):
            ctrl.add(tk.Radiobutton(ctrl, text=modo, variable=self._pp_elbow, value=modo,
                           bg="#0d1a0d", fg=TEXT, selectcolor=PANEL,
                           activebackground="#0d1a0d", activeforeground=ACCENT,
                           font=("Courier New", 9)))

        ctrl.add(tk.Frame(ctrl, bg=BORDER, width=2, height=30))

        # Selector de método de planificación por tramo
        ctrl.add(tk.Label(ctrl, text="Metodo:", bg="#0d1a0d", fg=ACCENT,
                          font=("Courier New", 9, "bold")))
        self._pp_metodo = tk.StringVar(value="articular")
        for lbl, val in [("Articular", "articular"),
                         ("Cartesiano", "cartesiano"),
                         ("Ctrl cinematico", "ctrl_cinematico")]:
            ctrl.add(tk.Radiobutton(ctrl, text=lbl, variable=self._pp_metodo, value=val,
                           bg="#0d1a0d", fg=TEXT, selectcolor=PANEL,
                           activebackground="#0d1a0d", activeforeground=ACCENT,
                           font=("Courier New", 9)))

        # Campo K (ganancia para el control cinematico)
        frm_K, self._pp_K = make_labeled_entry(ctrl, "K", "10.0", width=4)
        frm_K.configure(bg="#0d1a0d")
        for child in frm_K.winfo_children():
            try: child.configure(bg="#0d1a0d")
            except tk.TclError: pass
        ctrl.add(frm_K)

        ctrl.add(tk.Frame(ctrl, bg=BORDER, width=2, height=30))

        # ── Gripper ──
        ctrl.add(tk.Label(ctrl, text="Grip:", bg="#0d1a0d", fg=ACCENT,
                          font=("Courier New", 9, "bold")))
        self._pp_grip_en = tk.BooleanVar(value=False)
        ctrl.add(tk.Checkbutton(ctrl, text="activar", variable=self._pp_grip_en,
                                bg="#0d1a0d", fg=TEXT, selectcolor=PANEL,
                                activebackground="#0d1a0d", activeforeground=ACCENT,
                                font=("Courier New", 9), cursor="hand2"))
        self._pp_grip_pick  = tk.StringVar(value="cerrar")
        self._pp_grip_place = tk.StringVar(value="abrir")
        ctrl.add(tk.Label(ctrl, text="recogida:", bg="#0d1a0d", fg=MUTED,
                          font=("Courier New", 9)))
        for lbl, val in [("◯", "abrir"), ("⬤", "cerrar")]:
            ctrl.add(tk.Radiobutton(ctrl, text=lbl, variable=self._pp_grip_pick, value=val,
                           bg="#0d1a0d", fg=TEXT, selectcolor=PANEL,
                           activebackground="#0d1a0d", activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2"))
        ctrl.add(tk.Label(ctrl, text="depósito:", bg="#0d1a0d", fg=MUTED,
                          font=("Courier New", 9)))
        for lbl, val in [("◯", "abrir"), ("⬤", "cerrar")]:
            ctrl.add(tk.Radiobutton(ctrl, text=lbl, variable=self._pp_grip_place, value=val,
                           bg="#0d1a0d", fg=TEXT, selectcolor=PANEL,
                           activebackground="#0d1a0d", activeforeground=ACCENT,
                           font=("Courier New", 9), cursor="hand2"))

        ctrl.add(tk.Frame(ctrl, bg=BORDER, width=2, height=30))

        # Botón auxiliar: cargar como keyframes en pestaña 3
        ctrl.add(tk.Button(ctrl, text="  ↗ Cargar y animar en pestaña 3  ",
                  bg=BTN_HOV, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=12, pady=6,
                  command=self._pingpong_load_and_animate))

        # Estado
        self._lbl_pp_status = tk.Label(
            ctrl, text="Edita la lista abajo y presiona 'GENERAR CSV' para exportar.",
            bg="#0d1a0d", fg="#5a8a5a",
            font=("Courier New", 9))
        ctrl.add(self._lbl_pp_status)

        ctrl.finish()

        # ── Tabla de waypoints (editor) ──
        body = tk.Frame(parent, bg=BG)
        body.pack(fill="both", expand=True, side="top")

        # Panel izquierdo: tabla
        left = tk.Frame(body, bg=PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        title_row = tk.Frame(left, bg=PANEL)
        title_row.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(title_row, text="WAYPOINTS — editables (clic en X/Y/Z para modificar)",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(side="left")

        # Botones de utilidad
        tk.Button(title_row, text="  ↺ Restaurar 22 default  ",
                  bg=BTN_RSET, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=8, pady=3,
                  command=self._pp_restore_default).pack(side="right", padx=4)
        tk.Button(title_row, text="  ✖ Limpiar todo  ",
                  bg=BTN_BG, fg=MUTED,
                  relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=8, pady=3,
                  command=self._pp_clear_all).pack(side="right", padx=4)

        # Header de la tabla (anchos fijos para alinear con las filas)
        hdr_row = tk.Frame(left, bg="#0a0e14")
        hdr_row.pack(fill="x", padx=10)
        for col, w in (("#", 4), ("nombre", 12),
                       ("tipo", 5),
                       ("v1 (X/q1)", 9), ("v2 (Y/q2)", 9), ("v3 (Z/q3)", 9),
                       ("descripción", 22), ("acciones", 10)):
            tk.Label(hdr_row, text=col, width=w, anchor="w",
                     bg="#0a0e14", fg=MUTED,
                     font=("Courier New", 9, "bold")).pack(side="left", padx=2)

        # Filas — scrollable con un canvas + frame interno
        wrap = tk.Frame(left, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=10, pady=(2, 4))
        self._pp_tbl_canvas = tk.Canvas(wrap, bg=PANEL, highlightthickness=0,
                                        bd=0, height=200)   # solo altura minima
        self._pp_tbl_canvas.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(wrap, orient="vertical",
                          command=self._pp_tbl_canvas.yview)
        sb.pack(side="right", fill="y")
        self._pp_tbl_canvas.configure(yscrollcommand=sb.set)

        self._pp_tbl_inner = tk.Frame(self._pp_tbl_canvas, bg=PANEL)
        self._pp_tbl_canvas.create_window((0, 0), window=self._pp_tbl_inner,
                                          anchor="nw")
        self._pp_tbl_inner.bind(
            "<Configure>",
            lambda e: self._pp_tbl_canvas.configure(
                scrollregion=self._pp_tbl_canvas.bbox("all")))
        # Scroll con la rueda del mouse cuando el cursor esta sobre la tabla
        def _on_mousewheel(e):
            # En Linux las "rotaciones" llegan como Button-4 / Button-5
            delta = -1 if (e.delta < 0 or getattr(e, "num", 0) == 5) else 1
            self._pp_tbl_canvas.yview_scroll(-delta, "units")
        self._pp_tbl_canvas.bind("<MouseWheel>", _on_mousewheel)
        self._pp_tbl_canvas.bind("<Button-4>",   _on_mousewheel)
        self._pp_tbl_canvas.bind("<Button-5>",   _on_mousewheel)

        # ── Fila para AGREGAR un nuevo waypoint al final ──
        add_row = tk.Frame(left, bg="#0d1a0d",
                           highlightbackground=BORDER, highlightthickness=1)
        add_row.pack(fill="x", padx=10, pady=(2, 10))
        tk.Label(add_row, text="Agregar nuevo:", bg="#0d1a0d", fg="#f0c040",
                 font=("Courier New", 9, "bold")).pack(side="left", padx=(8, 6),
                                                        pady=6)
        # Campos del formulario de "nuevo waypoint"
        self._pp_add_name = tk.StringVar(value="WP")
        self._pp_add_x    = tk.StringVar(value="0.0")
        self._pp_add_y    = tk.StringVar(value="0.0")
        self._pp_add_z    = tk.StringVar(value="80.0")
        self._pp_add_desc = tk.StringVar(value="custom")
        self._pp_add_tipo = tk.StringVar(value="xyz")
        for var, w in [(self._pp_add_name, 11),
                       (self._pp_add_x,     8),
                       (self._pp_add_y,     8),
                       (self._pp_add_z,     8),
                       (self._pp_add_desc, 21)]:
            e = tk.Entry(add_row, textvariable=var, width=w,
                         bg="#0a0e14", fg=ACCENT, insertbackground=ACCENT,
                         relief="flat", font=("Courier New", 9),
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=BTN_HOV)
            e.pack(side="left", padx=2)
            # Enter en cualquier campo agrega el waypoint
            e.bind("<Return>", lambda _e: self._pp_add_waypoint())
        # Toggle XYZ / ANG para el formulario de agregar
        _add_tipo_btn = tk.Button(
            add_row, text="XYZ",
            bg="#1f4a3f", fg="#3fb950",
            relief="flat", bd=0, width=4,
            font=("Courier New", 8, "bold"), cursor="hand2",
            padx=2, pady=1)
        _add_tipo_btn.pack(side="left", padx=4)
        def _toggle_add_tipo(btn=_add_tipo_btn):
            if self._pp_add_tipo.get() == "xyz":
                self._pp_add_tipo.set("ang")
                btn.config(text="ANG", bg="#1f2a4a", fg="#58a6ff")
            else:
                self._pp_add_tipo.set("xyz")
                btn.config(text="XYZ", bg="#1f4a3f", fg="#3fb950")
        _add_tipo_btn.config(command=_toggle_add_tipo)
        tk.Button(add_row, text="  + Agregar al final  ",
                  bg=BTN_PLAY, fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 9, "bold"), cursor="hand2",
                  padx=10, pady=3,
                  command=self._pp_add_waypoint).pack(side="left", padx=8)

        # Renderizar la tabla por primera vez
        self._pp_render_table()

        # Panel derecho: resumen en tarjetas
        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="y", padx=(5, 10), pady=10)

        tk.Label(right, text="RESUMEN", bg=BG, fg=ACCENT,
                 font=("Courier New", 11, "bold")).pack(anchor="w", pady=(0, 8))

        # Tarjetas dinamicas (numero de waypoints y tramos cambian con la edicion)
        self._pp_card_wps = self._pp_make_card(right, "Waypoints totales",
                                               "22", "#58a6ff")
        self._pp_card_seg = self._pp_make_card(right, "Tramos",
                                               "21", "#3fb950")
        self._pp_make_card(right, "Z recogida (sugerido)", "40 mm", "#f78166")
        self._pp_make_card(right, "Z tránsito (sugerido)", "80 mm", "#f0c040")

        # Tiempo total (depende del valor actual de seg/tramo)
        self._card_tf = tk.Frame(right, bg=PANEL,
                                 highlightbackground=BORDER, highlightthickness=1)
        self._card_tf.pack(fill="x", pady=4, ipadx=10, ipady=8)
        tk.Label(self._card_tf, text="Tiempo total", bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="w", padx=8)
        self._lbl_card_tf = tk.Label(
            self._card_tf, text="31.5 s",
            bg=PANEL, fg=ACCENT,
            font=("Courier New", 14, "bold"))
        self._lbl_card_tf.pack(anchor="w", padx=8)
        self._lbl_card_tf_sub = tk.Label(
            self._card_tf, text="a 1.5 s/tramo",
            bg=PANEL, fg=MUTED,
            font=("Courier New", 8))
        self._lbl_card_tf_sub.pack(anchor="w", padx=8)

        # Actualizar la tarjeta de tiempo total cuando cambie seg/tramo
        def _update_tf_card(*_):
            try:
                s = float(self._pp_dur_var.get())
                n_seg = max(0, len(self._pp_waypoints) - 1)
                self._lbl_card_tf.config(text=f"{s * n_seg:.1f} s")
                self._lbl_card_tf_sub.config(text=f"a {s:.2f} s/tramo")
            except (ValueError, tk.TclError):
                self._lbl_card_tf.config(text="—")
        self._pp_update_tf_card = _update_tf_card   # exponer para llamar desde edicion
        self._pp_dur_var.trace_add("write", _update_tf_card)

    # ─────────────────────────────────────────────────────────────────
    #  EDITOR DE WAYPOINTS DEL PING PONG
    # ─────────────────────────────────────────────────────────────────
    def _pp_make_card(self, parent, title, value, color):
        """Crea una tarjeta de resumen y devuelve la Label de valor (mutable)."""
        card = tk.Frame(parent, bg=PANEL,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=4, ipadx=10, ipady=8)
        tk.Label(card, text=title, bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="w", padx=8)
        lbl = tk.Label(card, text=value, bg=PANEL, fg=color,
                       font=("Courier New", 14, "bold"))
        lbl.pack(anchor="w", padx=8)
        return lbl

    def _pp_color_for_name(self, name):
        """Color de la columna 'nombre' segun el tipo de waypoint."""
        if "REPOSO" in name:
            return "#b392f0"
        if name and name[0].islower():        # p1, p2, p3, p1_up, ...
            return "#3fb950" if not name.endswith("_up") else "#2d6e3a"
        if name and name[0].isupper():        # P1, P2, P3, ...
            return "#58a6ff" if not name.endswith("_up") else "#1f4a80"
        return TEXT

    def _pp_render_table(self):
        """Reconstruye la tabla de waypoints a partir de self._pp_waypoints.
        Llamada cuando cambia el numero de filas (insertar/borrar/restaurar)."""
        # Asegurar inicializacion
        if not hasattr(self, "_pp_waypoints") or self._pp_waypoints is None:
            self._pp_waypoints = self._pp_default_waypoints()

        # Limpiar contenido anterior
        for w in self._pp_tbl_inner.winfo_children():
            w.destroy()

        # Lista para mantener referencias a las StringVar de cada fila
        # (necesario para que los trace no se garbage-collecten)
        self._pp_row_vars = []

        for i, wp in enumerate(self._pp_waypoints):
            self._pp_make_row(i, wp)

        # Actualizar tarjetas de resumen
        n = len(self._pp_waypoints)
        if hasattr(self, "_pp_card_wps"):
            self._pp_card_wps.config(text=str(n))
        if hasattr(self, "_pp_card_seg"):
            self._pp_card_seg.config(text=str(max(0, n - 1)))
        if hasattr(self, "_pp_update_tf_card"):
            try: self._pp_update_tf_card()
            except Exception: pass
        # scrollregion
        self._pp_tbl_inner.update_idletasks()
        try:
            self._pp_tbl_canvas.configure(
                scrollregion=self._pp_tbl_canvas.bbox("all"))
        except tk.TclError:
            pass

    def _pp_make_row(self, idx, wp):
        """Crea una fila editable para el waypoint en la posicion idx.
        wp puede tener 3 o 4 elementos: [nombre, [v1,v2,v3], desc] o [..., tipo]
        tipo = 'xyz' → v1,v2,v3 son X,Y,Z en mm
        tipo = 'ang' → v1,v2,v3 son q1,q2,q3 en grados
        """
        name = wp[0]
        vals = wp[1]
        desc = wp[2]
        tipo = wp[3] if len(wp) > 3 else "xyz"
        # Asegurar que el waypoint tiene el campo tipo
        if len(wp) == 3:
            wp.append("xyz")
            tipo = "xyz"

        bg_row = "#161b22" if idx % 2 == 0 else "#0d1117"
        row = tk.Frame(self._pp_tbl_inner, bg=bg_row)
        row.pack(fill="x", pady=0)

        # # (indice — solo lectura)
        tk.Label(row, text=str(idx), width=4, anchor="w",
                 bg=bg_row, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=2)

        # Nombre (editable)
        name_var = tk.StringVar(value=name)
        e_name = tk.Entry(row, textvariable=name_var, width=11,
                          bg=bg_row, fg=self._pp_color_for_name(name),
                          insertbackground=ACCENT,
                          relief="flat", font=("Courier New", 9, "bold"),
                          highlightthickness=1, highlightbackground=bg_row,
                          highlightcolor=BTN_HOV)
        e_name.pack(side="left", padx=2)

        # ── Toggle de tipo (XYZ / ANG) ──
        tipo_var = tk.StringVar(value=tipo)

        # Colores para los estados del toggle
        def _tipo_colors(t):
            if t == "xyz":
                return "#1f4a3f", "#3fb950"   # bg, fg
            else:
                return "#1f2a4a", "#58a6ff"

        btn_tipo_bg, btn_tipo_fg = _tipo_colors(tipo)
        btn_tipo = tk.Button(row, text="XYZ" if tipo == "xyz" else "ANG",
                             bg=btn_tipo_bg, fg=btn_tipo_fg,
                             relief="flat", bd=0, width=4,
                             font=("Courier New", 8, "bold"), cursor="hand2",
                             padx=2, pady=1)
        btn_tipo.pack(side="left", padx=2)

        # Variables para los tres valores
        v1_var = tk.StringVar(value=f"{vals[0]:.2f}")
        v2_var = tk.StringVar(value=f"{vals[1]:.2f}")
        v3_var = tk.StringVar(value=f"{vals[2]:.2f}")

        # Labels dinámicas que se actualizan con el tipo
        lbl_colors = {"xyz": MUTED, "ang": "#b392f0"}

        entries_v = []
        for var in (v1_var, v2_var, v3_var):
            e = tk.Entry(row, textvariable=var, width=8,
                         bg="#0a0e14", fg=TEXT, insertbackground=ACCENT,
                         relief="flat", font=("Courier New", 9),
                         highlightthickness=1, highlightbackground=bg_row,
                         highlightcolor=BTN_HOV)
            e.pack(side="left", padx=2)
            entries_v.append(e)

        def _toggle_tipo(i=idx, tv=tipo_var, btn=btn_tipo, evs=entries_v,
                         v1=v1_var, v2=v2_var, v3=v3_var):
            """Alterna el tipo entre 'xyz' y 'ang'. Si el tipo cambia y los
            valores son válidos, convierte automáticamente usando FK o IK."""
            old_t = tv.get()
            new_t = "ang" if old_t == "xyz" else "xyz"
            tv.set(new_t)
            # Actualizar botón
            bg_c, fg_c = _tipo_colors(new_t)
            btn.config(text="XYZ" if new_t == "xyz" else "ANG",
                       bg=bg_c, fg=fg_c)
            # Intentar conversión automática de los valores
            try:
                a = float(v1.get())
                b = float(v2.get())
                c = float(v3.get())
                if old_t == "xyz" and new_t == "ang":
                    # xyz → ang: usar IK
                    sol = ik(a, b, c, elbow_up=True)
                    if sol is not None:
                        v1.set(f"{sol[0]:.2f}")
                        v2.set(f"{sol[1]:.2f}")
                        v3.set(f"{sol[2]:.2f}")
                elif old_t == "ang" and new_t == "xyz":
                    # ang → xyz: usar FK
                    xyz = ee_pos(np.array([a, b, c], dtype=float))
                    v1.set(f"{xyz[0]:.2f}")
                    v2.set(f"{xyz[1]:.2f}")
                    v3.set(f"{xyz[2]:.2f}")
            except (ValueError, TypeError):
                pass
            # Guardar en la lista
            try:
                if i < len(self._pp_waypoints):
                    self._pp_waypoints[i][3] = new_t
            except IndexError:
                pass

        btn_tipo.config(command=_toggle_tipo)

        # Descripcion (editable)
        desc_var = tk.StringVar(value=desc)
        e_desc = tk.Entry(row, textvariable=desc_var, width=22,
                          bg=bg_row, fg=MUTED, insertbackground=ACCENT,
                          relief="flat", font=("Courier New", 9),
                          highlightthickness=1, highlightbackground=bg_row,
                          highlightcolor=BTN_HOV)
        e_desc.pack(side="left", padx=2)

        # Botones de accion (eliminar / insertar abajo)
        tk.Button(row, text="✕", width=2,
                  bg="#7d1f1f", fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 8, "bold"), cursor="hand2",
                  command=lambda i=idx: self._pp_delete_waypoint(i)
                  ).pack(side="left", padx=2)
        tk.Button(row, text="↑+", width=3,
                  bg="#1f4a3f", fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 8, "bold"), cursor="hand2",
                  command=lambda i=idx: self._pp_insert_above(i)
                  ).pack(side="left", padx=2)
        tk.Button(row, text="↓+", width=3,
                  bg="#1f4a3f", fg=ACCENT,
                  relief="flat", bd=0,
                  font=("Courier New", 8, "bold"), cursor="hand2",
                  command=lambda i=idx: self._pp_insert_below(i)
                  ).pack(side="left", padx=2)

        # Conectar trace para actualizar self._pp_waypoints al editar.
        def _on_change(*_args, i=idx,
                       n=name_var, v1=v1_var, v2=v2_var, v3=v3_var,
                       d=desc_var, tv=tipo_var):
            try:
                a = float(v1.get())
                b = float(v2.get())
                c = float(v3.get())
                self._pp_waypoints[i][0] = n.get()
                self._pp_waypoints[i][1] = [a, b, c]
                self._pp_waypoints[i][2] = d.get()
                if len(self._pp_waypoints[i]) > 3:
                    self._pp_waypoints[i][3] = tv.get()
                else:
                    self._pp_waypoints[i].append(tv.get())
                # Refrescar color del nombre
                e_name.config(fg=self._pp_color_for_name(n.get()))
            except (ValueError, IndexError):
                pass
        for v in (name_var, v1_var, v2_var, v3_var, desc_var):
            v.trace_add("write", _on_change)

        # Guardar las StringVar para que no las recoja el GC
        self._pp_row_vars.append((name_var, v1_var, v2_var, v3_var, desc_var, tipo_var))

    # ── Operaciones del editor ──
    def _pp_add_waypoint(self):
        """Agrega el waypoint actualmente en los campos 'Agregar nuevo' al final."""
        try:
            v1 = float(self._pp_add_x.get())
            v2 = float(self._pp_add_y.get())
            v3 = float(self._pp_add_z.get())
        except ValueError:
            tipo_actual = getattr(self, "_pp_add_tipo", None)
            tipo_str = tipo_actual.get() if tipo_actual else "xyz"
            lbl = ("X, Y y Z (mm)" if tipo_str == "xyz"
                   else "q1, q2, q3 (grados)")
            messagebox.showerror(
                "Valores invalidos",
                f"{lbl} deben ser numeros validos.")
            return
        name = self._pp_add_name.get().strip() or f"WP{len(self._pp_waypoints)}"
        desc = self._pp_add_desc.get().strip()
        tipo = (self._pp_add_tipo.get()
                if hasattr(self, "_pp_add_tipo") else "xyz")
        self._pp_waypoints.append([name, [v1, v2, v3], desc, tipo])
        self._pp_render_table()
        # Auto-scroll al final para que el usuario vea el nuevo
        try:
            self._pp_tbl_canvas.update_idletasks()
            self._pp_tbl_canvas.yview_moveto(1.0)
        except tk.TclError:
            pass
        tipo_label = "XYZ (mm)" if tipo == "xyz" else "ANG (grados)"
        self._lbl_pp_status.config(
            text=f"+ Waypoint '{name}' [{tipo_label}] agregado al final ({len(self._pp_waypoints)} totales).",
            fg="#3fb950")

    def _pp_delete_waypoint(self, idx):
        """Elimina la fila idx (con confirmacion si esta cerca de los limites)."""
        if idx < 0 or idx >= len(self._pp_waypoints):
            return
        if len(self._pp_waypoints) <= 2:
            messagebox.showwarning(
                "Minimo requerido",
                "Se necesitan al menos 2 waypoints para definir una trayectoria.\n"
                "No se puede borrar mas.")
            return
        name = self._pp_waypoints[idx][0]
        del self._pp_waypoints[idx]
        self._pp_render_table()
        self._lbl_pp_status.config(
            text=f"− Waypoint #{idx} ('{name}') eliminado ({len(self._pp_waypoints)} restantes).",
            fg="#f0c040")

    def _pp_insert_above(self, idx):
        """Inserta una copia del waypoint idx justo encima de el."""
        if idx < 0 or idx >= len(self._pp_waypoints):
            return
        wp = self._pp_waypoints[idx]
        nm  = wp[0]; vals = wp[1]; desc = wp[2]
        tipo = wp[3] if len(wp) > 3 else "xyz"
        self._pp_waypoints.insert(idx, [nm, list(vals), desc, tipo])
        self._pp_render_table()
        self._lbl_pp_status.config(
            text=f"↑+ Inserto copia de '{nm}' arriba (#{idx}).", fg="#3fb950")

    def _pp_insert_below(self, idx):
        """Inserta una copia del waypoint idx justo debajo de el."""
        if idx < 0 or idx >= len(self._pp_waypoints):
            return
        wp = self._pp_waypoints[idx]
        nm  = wp[0]; vals = wp[1]; desc = wp[2]
        tipo = wp[3] if len(wp) > 3 else "xyz"
        self._pp_waypoints.insert(idx + 1, [nm, list(vals), desc, tipo])
        self._pp_render_table()
        self._lbl_pp_status.config(
            text=f"↓+ Inserto copia de '{nm}' abajo (#{idx+1}).", fg="#3fb950")

    def _pp_clear_all(self):
        """Borra todos los waypoints (deja una pose REPOSO minima)."""
        if not messagebox.askyesno(
                "Limpiar todo",
                f"¿Eliminar TODOS los {len(self._pp_waypoints)} waypoints?\n\n"
                "Quedara solo el waypoint REPOSO inicial. Puedes restaurar los "
                "22 default con el boton '↺ Restaurar 22 default'."):
            return
        self._pp_waypoints = [["REPOSO", list(self._PP_REPOSO), "inicio", "xyz"]]
        self._pp_render_table()
        self._lbl_pp_status.config(
            text="Lista limpiada — solo queda 1 waypoint.", fg="#f0c040")

    def _pp_restore_default(self):
        """Restaura los 22 waypoints predeterminados."""
        if (hasattr(self, "_pp_waypoints") and self._pp_waypoints
                and len(self._pp_waypoints) > 0):
            if not messagebox.askyesno(
                    "Restaurar default",
                    f"¿Reemplazar los {len(self._pp_waypoints)} waypoints actuales "
                    "con la secuencia predeterminada de 22?"):
                return
        self._pp_waypoints = self._pp_default_waypoints()
        self._pp_render_table()
        self._lbl_pp_status.config(
            text="Restaurados los 22 waypoints predeterminados.", fg="#3fb950")

    def _pingpong_build_keyframes_q(self, elbow_up=True):
        """Resuelve IK para waypoints cartesianos; usa q directamente para tipo 'ang'.
        Devuelve (keyframes_q, fallos_ik, advertencias_rango).
          fallos_ik        — waypoints XYZ sin solución IK (bloqueante).
          advertencias_rango — waypoints ANG fuera de Q_MIN/Q_MAX (no bloqueante).
        """
        if not hasattr(self, "_pp_waypoints") or self._pp_waypoints is None:
            self._pp_waypoints = self._pp_default_waypoints()
        keyframes_q   = []
        fallos        = []
        advertencias  = []
        for i, wp in enumerate(self._pp_waypoints):
            name = wp[0]
            vals = wp[1]
            tipo = wp[3] if len(wp) > 3 else "xyz"
            if tipo == "ang":
                q = np.array(vals, dtype=float)
                # Siempre se incluye como keyframe; si está fuera de rango
                # se avisa pero NO se bloquea.
                if np.any(q < Q_MIN - 0.5) or np.any(q > Q_MAX + 0.5):
                    advertencias.append((i, name, vals))
                keyframes_q.append(q)
            else:
                x, y, z = vals[0], vals[1], vals[2]
                q = ik(x, y, z, elbow_up=elbow_up)
                if q is None:
                    fallos.append((i, name, (x, y, z)))
                else:
                    keyframes_q.append(q)
        return keyframes_q, fallos, advertencias

    def _pingpong_build_trajectory(self, keyframes_q, tf_seg,
                                   metodo="articular",
                                   keyframes_xyz=None,
                                   K_gain=10.0,
                                   elbow_up=True):
        """Construye la trayectoria tramo a tramo concatenando los keyframes.

        Métodos soportados:
          - "articular":       perfil quíntico sobre q1, q2, q3 (curva en XYZ)
          - "cartesiano":      perfil quíntico sobre X, Y, Z + IK por punto
          - "ctrl_cinematico": x_d quíntico en XYZ + control con J⁻¹(q)(ẋ_d + Ke)

        Para "cartesiano" y "ctrl_cinematico" se requiere `keyframes_xyz`.
        Devuelve (t, q, qd, qdd, ee) como arrays.
        """
        t_global   = []
        q_global   = [[], [], []]
        qd_global  = [[], [], []]
        qdd_global = [[], [], []]
        ee_global  = [[], [], []]
        t_offset = 0.0

        # Estado articular que va arrastrando entre tramos en modo control
        # cinemático (la pose final efectiva del tramo previo es la inicial
        # del siguiente; así no se reinicia el lazo y se preserva la continuidad).
        q_state = keyframes_q[0].copy()

        for seg in range(len(keyframes_q) - 1):
            q0_seg = keyframes_q[seg]
            qf_seg = keyframes_q[seg + 1]
            t_local = np.arange(0.0, tf_seg + DT, DT)
            # En tramos posteriores al primero, descartar t=0 para no duplicar
            # el punto frontera con el tramo anterior.
            drop_first = (seg > 0)
            t_use = t_local[1:] if drop_first else t_local

            if metodo == "articular":
                # ---- Perfil quíntico articular ----
                q_seg, qd_seg, qdd_seg = perfil_quintico(
                    q0_seg, qf_seg, t_local, tf_seg)
                if drop_first:
                    q_seg   = q_seg[:, 1:]
                    qd_seg  = qd_seg[:, 1:]
                    qdd_seg = qdd_seg[:, 1:]
                ee_seg = np.array([ee_pos(q_seg[:, i])
                                   for i in range(q_seg.shape[1])]).T  # (3, n)
                # Actualizar estado articular
                q_state = q_seg[:, -1].copy()

            elif metodo == "cartesiano":
                # ---- Perfil quíntico cartesiano + IK por punto ----
                if keyframes_xyz is None:
                    raise ValueError("keyframes_xyz requerido en modo cartesiano")
                p0_seg = np.asarray(keyframes_xyz[seg],     dtype=float)
                pf_seg = np.asarray(keyframes_xyz[seg + 1], dtype=float)
                ee_seg = np.zeros((3, len(t_local)))
                for k in range(3):
                    D = pf_seg[k] - p0_seg[k]
                    a3 =  10.0 * D / tf_seg**3
                    a4 = -15.0 * D / tf_seg**4
                    a5 =   6.0 * D / tf_seg**5
                    ee_seg[k] = (p0_seg[k] + a3*t_local**3
                                 + a4*t_local**4 + a5*t_local**5)
                # IK punto a punto
                n_pts = len(t_local)
                q_seg = np.zeros((3, n_pts))
                q_prev = q_state.copy()
                for i in range(n_pts):
                    sol = ik(ee_seg[0, i], ee_seg[1, i], ee_seg[2, i],
                             elbow_up=elbow_up)
                    q_seg[:, i] = sol if sol is not None else q_prev
                    q_prev = q_seg[:, i]
                qd_seg  = np.gradient(q_seg, t_local, axis=1)
                qdd_seg = np.gradient(qd_seg, t_local, axis=1)
                if drop_first:
                    q_seg   = q_seg[:, 1:]
                    qd_seg  = qd_seg[:, 1:]
                    qdd_seg = qdd_seg[:, 1:]
                    ee_seg  = ee_seg[:, 1:]
                q_state = q_seg[:, -1].copy()

            else:
                # ---- Control cinemático con jacobiano ----
                if keyframes_xyz is None:
                    raise ValueError("keyframes_xyz requerido en modo ctrl_cinematico")
                p0_seg = np.asarray(keyframes_xyz[seg],     dtype=float)
                pf_seg = np.asarray(keyframes_xyz[seg + 1], dtype=float)
                # Trayectoria deseada en XYZ (perfil quíntico) y su derivada
                xd     = np.zeros((len(t_local), 3))
                xd_dot = np.zeros((len(t_local), 3))
                for k in range(3):
                    D = pf_seg[k] - p0_seg[k]
                    a3 =  10.0 * D / tf_seg**3
                    a4 = -15.0 * D / tf_seg**4
                    a5 =   6.0 * D / tf_seg**5
                    xd[:,     k] = (p0_seg[k] + a3*t_local**3
                                    + a4*t_local**4 + a5*t_local**5)
                    xd_dot[:,  k] = (3*a3*t_local**2 + 4*a4*t_local**3
                                     + 5*a5*t_local**4)

                # Importante: q_state arrastrado del tramo previo, NO el q_seg
                # del IK puro. Así el error inicial del tramo nuevo es e=xd[0]-x_e
                # (cero si el tramo previo terminó bien).
                q_seg, qd_seg, qdd_seg, pos_ee_seg, _err = control_cinematico(
                    q0_deg=q_state, xd=xd, xd_dot=xd_dot, t=t_local,
                    K_gain=K_gain, dls_lambda=0.01)
                ee_seg = pos_ee_seg.T   # (3, n)
                if drop_first:
                    q_seg   = q_seg[:, 1:]
                    qd_seg  = qd_seg[:, 1:]
                    qdd_seg = qdd_seg[:, 1:]
                    ee_seg  = ee_seg[:, 1:]
                q_state = q_seg[:, -1].copy()

            # Acumular
            for j in range(3):
                q_global[j].extend(q_seg[j].tolist())
                qd_global[j].extend(qd_seg[j].tolist())
                qdd_global[j].extend(qdd_seg[j].tolist())
                ee_global[j].extend(ee_seg[j].tolist())
            t_global.extend((t_use + t_offset).tolist())
            t_offset += tf_seg

        t_arr   = np.array(t_global)
        q_arr   = np.array(q_global)
        qd_arr  = np.array(qd_global)
        qdd_arr = np.array(qdd_global)
        ee_arr  = np.array(ee_global).T   # (N, 3)
        return t_arr, q_arr, qd_arr, qdd_arr, ee_arr

    def _pingpong_generate_csv(self):
        """UN SOLO CLIC: resuelve IK de los 22 waypoints, construye la trayectoria
        tramo a tramo con el método elegido y exporta el CSV."""
        # Leer parámetros
        try:
            tf_seg = float(self._pp_dur_var.get())
            if tf_seg <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Duración inválida",
                                 "Ingresa un valor positivo en 'seg/tramo'.")
            return
        elbow_up = (self._pp_elbow.get() == "arriba")
        metodo   = self._pp_metodo.get()

        # Leer K solo si se usa control cinemático
        K_gain = 10.0
        if metodo == "ctrl_cinematico":
            try:
                K_gain = float(self._pp_K.get())
                if K_gain <= 0:
                    raise ValueError
            except (ValueError, tk.TclError):
                messagebox.showerror(
                    "K invalida",
                    "La ganancia K del control cinematico debe ser un numero positivo.\n"
                    "Valores tipicos: 5 - 50.")
                return

        # Resolver IK de todos los waypoints (siempre, para keyframes_q de respaldo
        # y para el modo articular puro)
        keyframes_q, fallos, advertencias = self._pingpong_build_keyframes_q(elbow_up=elbow_up)
        if fallos:
            msg = "Los siguientes waypoints XYZ no tienen solución IK y fueron omitidos:\n\n"
            for i, name, (x, y, z) in fallos:
                msg += f"  #{i:2d} {name}:  ({x:+.1f}, {y:+.1f}, {z:+.1f})\n"
            msg += "\nRevisa los parámetros DH del robot o cambia el modo de codo."
            messagebox.showerror("IK fuera de alcance", msg)
            return
        if advertencias:
            msg = "⚠ Los siguientes waypoints ANG están fuera del rango articular físico:\n\n"
            for i, name, vals in advertencias:
                limites = []
                q = np.array(vals, dtype=float)
                for j, (qj, lo, hi, nm) in enumerate(zip(q, Q_MIN, Q_MAX,
                                                          ["q1", "q2", "q3"])):
                    if qj < lo - 0.5 or qj > hi + 0.5:
                        limites.append(f"{nm}={qj:.1f}° (rango [{lo:.0f}°, {hi:.0f}°])")
                msg += f"  #{i:2d} {name}:  {',  '.join(limites)}\n"
            msg += "\n¿Deseas continuar de todos modos?\n(El hardware podría alcanzar límites físicos.)"
            if not messagebox.askyesno("Advertencia de rango articular", msg,
                                       icon="warning"):
                return
        if len(keyframes_q) < 2:
            messagebox.showerror("Error", "No se pudieron generar los keyframes.")
            return

        # Lista de waypoints cartesianos en el mismo orden
        wps = self._pp_waypoint_list()
        keyframes_xyz = [np.array(xyz, dtype=float) for (_, xyz, _) in wps]

        # Construir trayectoria
        t_arr, q_arr, qd_arr, qdd_arr, ee_arr = self._pingpong_build_trajectory(
            keyframes_q, tf_seg,
            metodo=metodo, keyframes_xyz=keyframes_xyz,
            K_gain=K_gain, elbow_up=elbow_up)
        N = len(t_arr)

        # Diálogo para guardar
        default_name = f"trayectoria_pingpong_{metodo}.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name,
            title="Guardar trayectoria ping pong — CSV")
        if not filepath:
            self._lbl_pp_status.config(
                text="Exportación cancelada.", fg="#f0c040")
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_s",
                "q1_deg", "q2_deg", "q3_deg",
                "q1d_dps", "q2d_dps", "q3d_dps",
                "q1dd_dps2", "q2dd_dps2", "q3dd_dps2",
                "ee_x_mm", "ee_y_mm", "ee_z_mm",
            ])
            for i in range(N):
                writer.writerow([
                    f"{t_arr[i]:.4f}",
                    f"{q_arr[0,i]:.4f}", f"{q_arr[1,i]:.4f}", f"{q_arr[2,i]:.4f}",
                    f"{qd_arr[0,i]:.4f}", f"{qd_arr[1,i]:.4f}", f"{qd_arr[2,i]:.4f}",
                    f"{qdd_arr[0,i]:.4f}", f"{qdd_arr[1,i]:.4f}", f"{qdd_arr[2,i]:.4f}",
                    f"{ee_arr[i,0]:.4f}", f"{ee_arr[i,1]:.4f}", f"{ee_arr[i,2]:.4f}",
                ])

        # Etiqueta de método humanizada
        metodo_lbl = {
            "articular":       "ping pong (articular)",
            "cartesiano":      "ping pong (cartesiano + IK)",
            "ctrl_cinematico": f"ping pong (control cinematico K={K_gain:.1f})",
        }[metodo]

        # Para modo control cinemático, calcular el error cartesiano respecto a
        # la trayectoria deseada (línea recta tramo a tramo)
        err_ee_full = None
        xd_track    = None
        if metodo == "ctrl_cinematico":
            xd_track    = self._pp_build_xd_track(keyframes_xyz, tf_seg)
            err_ee_full = xd_track - ee_arr

        # Grip events: close at pickup waypoints, open at deposit waypoints
        pp_grip_events = []
        if self._pp_grip_en.get():
            pick_action  = self._pp_grip_pick.get()
            place_action = self._pp_grip_place.get()
            pp_grip_events.append((0.0, "abrir"))   # always start open
            for i, (name, xyz, desc) in enumerate(wps):
                t_arrive = i * tf_seg
                dl = desc.lower()
                if "recoger" in dl:
                    pp_grip_events.append((t_arrive, pick_action))
                elif "soltar" in dl:
                    pp_grip_events.append((t_arrive, place_action))

        # Dejar la trayectoria cargada en _tr_data para poder animar en pestaña 3
        self._tr_data = dict(
            t=t_arr, q=q_arr, qd=qd_arr, qdd=qdd_arr,
            q0=keyframes_q[0], qf=keyframes_q[-1],
            tf=t_arr[-1],
            p0=ee_arr[0], pf=ee_arr[-1],
            pos_ee=ee_arr,
            v_ee=np.gradient(ee_arr, t_arr, axis=0),
            speed_ee=np.linalg.norm(np.gradient(ee_arr, t_arr, axis=0), axis=1),
            metodo=metodo_lbl,
            err_ee=err_ee_full, K_used=K_gain if metodo == "ctrl_cinematico" else None,
            xd_track=xd_track,
            grip_events=pp_grip_events,
        )
        # Guardar los keyframes también en _rec_keyframes/_rec_times para que
        # la pestaña 3 los dibuje en la traza del efector final
        self._rec_keyframes = [q.copy() for q in keyframes_q]
        self._rec_times     = [i * tf_seg for i in range(len(keyframes_q))]

        tf_total = t_arr[-1]
        K_msg = f"\n  K (ctrl cin.): {K_gain:.2f}" if metodo == "ctrl_cinematico" else ""
        err_msg = ""
        if err_ee_full is not None:
            err_max = np.max(np.linalg.norm(err_ee_full, axis=1))
            err_msg = f"\n  |e|max       : {err_max:.3f} mm"

        messagebox.showinfo(
            "CSV generado",
            f"Trayectoria PING PONG exportada correctamente.\n\n"
            f"  Archivo      : {os.path.basename(filepath)}\n"
            f"  Waypoints    : {len(keyframes_q)}\n"
            f"  Tramos       : {len(keyframes_q)-1}  ({tf_seg:.2f} s/tramo)\n"
            f"  Metodo       : {metodo}{K_msg}{err_msg}\n"
            f"  Tiempo total : {tf_total:.2f} s\n"
            f"  Puntos (dt={DT}s): {N}\n"
            f"  Modo codo    : {'arriba' if elbow_up else 'abajo'}\n\n"
            f"Columnas: t, q1/q2/q3 (deg), velocidades (deg/s),\n"
            f"          aceleraciones (deg/s²), EE x/y/z (mm)\n\n"
            f"La trayectoria quedó cargada en la pestaña\n"
            f"'3. Trayectoria' por si quieres animarla."
        )
        self._lbl_pp_status.config(
            text=f"✅ CSV guardado: {os.path.basename(filepath)}  — "
                 f"{N} puntos, tf={tf_total:.2f}s  [{metodo}]",
            fg="#3fb950")
        self._lbl_status.config(
            text=f"✅ Ping pong CSV: {os.path.basename(filepath)}  "
                 f"— {len(keyframes_q)} waypoints, {N} puntos, "
                 f"tf={tf_total:.2f}s  [{metodo}]")

    def _pp_build_xd_track(self, keyframes_xyz, tf_seg):
        """Reconstruye x_d(t) concatenando los perfiles quínticos cartesianos
        tramo a tramo (mismo timing que _pingpong_build_trajectory)."""
        xd_global = [[], [], []]
        for seg in range(len(keyframes_xyz) - 1):
            p0_seg = np.asarray(keyframes_xyz[seg],     dtype=float)
            pf_seg = np.asarray(keyframes_xyz[seg + 1], dtype=float)
            t_local = np.arange(0.0, tf_seg + DT, DT)
            xd_seg = np.zeros((3, len(t_local)))
            for k in range(3):
                D = pf_seg[k] - p0_seg[k]
                a3 =  10.0 * D / tf_seg**3
                a4 = -15.0 * D / tf_seg**4
                a5 =   6.0 * D / tf_seg**5
                xd_seg[k] = (p0_seg[k] + a3*t_local**3
                             + a4*t_local**4 + a5*t_local**5)
            if seg > 0:
                xd_seg = xd_seg[:, 1:]
            for j in range(3):
                xd_global[j].extend(xd_seg[j].tolist())
        return np.array(xd_global).T   # (N, 3)

    def _pingpong_load_and_animate(self):
        """Carga los 22 waypoints, construye la trayectoria en memoria con el
        método elegido (sin CSV) y salta a la pestaña 3 para animarla."""
        try:
            tf_seg = float(self._pp_dur_var.get())
            if tf_seg <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Duración inválida",
                                 "Ingresa un valor positivo en 'seg/tramo'.")
            return
        elbow_up = (self._pp_elbow.get() == "arriba")
        metodo   = self._pp_metodo.get()

        K_gain = 10.0
        if metodo == "ctrl_cinematico":
            try:
                K_gain = float(self._pp_K.get())
                if K_gain <= 0:
                    raise ValueError
            except (ValueError, tk.TclError):
                messagebox.showerror(
                    "K invalida",
                    "La ganancia K del control cinematico debe ser un numero positivo.\n"
                    "Valores tipicos: 5 - 50.")
                return

        keyframes_q, fallos, advertencias = self._pingpong_build_keyframes_q(elbow_up=elbow_up)
        if fallos:
            msg = "Waypoints XYZ sin solución IK (omitidos):\n\n"
            for i, name, (x, y, z) in fallos:
                msg += f"  #{i:2d} {name}:  ({x:+.1f}, {y:+.1f}, {z:+.1f})\n"
            messagebox.showerror("IK fuera de alcance", msg)
            return
        if advertencias:
            msg = "⚠ Los siguientes waypoints ANG están fuera del rango articular físico:\n\n"
            for i, name, vals in advertencias:
                q = np.array(vals, dtype=float)
                limites = []
                for j, (qj, lo, hi, nm) in enumerate(zip(q, Q_MIN, Q_MAX,
                                                          ["q1", "q2", "q3"])):
                    if qj < lo - 0.5 or qj > hi + 0.5:
                        limites.append(f"{nm}={qj:.1f}° (rango [{lo:.0f}°, {hi:.0f}°])")
                msg += f"  #{i:2d} {name}:  {',  '.join(limites)}\n"
            msg += "\n¿Deseas continuar de todos modos?\n(El hardware podría alcanzar límites físicos.)"
            if not messagebox.askyesno("Advertencia de rango articular", msg,
                                       icon="warning"):
                return

        wps = self._pp_waypoint_list()
        keyframes_xyz = [np.array(xyz, dtype=float) for (_, xyz, _) in wps]

        t_arr, q_arr, qd_arr, qdd_arr, ee_arr = self._pingpong_build_trajectory(
            keyframes_q, tf_seg,
            metodo=metodo, keyframes_xyz=keyframes_xyz,
            K_gain=K_gain, elbow_up=elbow_up)

        metodo_lbl = {
            "articular":       "ping pong (articular)",
            "cartesiano":      "ping pong (cartesiano + IK)",
            "ctrl_cinematico": f"ping pong (control cinematico K={K_gain:.1f})",
        }[metodo]

        err_ee_full = None
        xd_track    = None
        if metodo == "ctrl_cinematico":
            xd_track    = self._pp_build_xd_track(keyframes_xyz, tf_seg)
            err_ee_full = xd_track - ee_arr

        # Grip events (mismo calculo que _pingpong_generate_csv)
        la_grip_events = []
        if self._pp_grip_en.get():
            pick_action  = self._pp_grip_pick.get()
            place_action = self._pp_grip_place.get()
            la_grip_events.append((0.0, "abrir"))
            for i, (name, xyz, desc) in enumerate(wps):
                t_arrive = i * tf_seg
                dl = desc.lower()
                if "recoger" in dl:
                    la_grip_events.append((t_arrive, pick_action))
                elif "soltar" in dl:
                    la_grip_events.append((t_arrive, place_action))

        self._tr_data = dict(
            t=t_arr, q=q_arr, qd=qd_arr, qdd=qdd_arr,
            q0=keyframes_q[0], qf=keyframes_q[-1],
            tf=t_arr[-1],
            p0=ee_arr[0], pf=ee_arr[-1],
            pos_ee=ee_arr,
            v_ee=np.gradient(ee_arr, t_arr, axis=0),
            speed_ee=np.linalg.norm(np.gradient(ee_arr, t_arr, axis=0), axis=1),
            metodo=metodo_lbl,
            err_ee=err_ee_full, K_used=K_gain if metodo == "ctrl_cinematico" else None,
            xd_track=xd_track,
            grip_events=la_grip_events,
        )
        self._rec_keyframes = [q.copy() for q in keyframes_q]
        self._rec_times     = [i * tf_seg for i in range(len(keyframes_q))]

        self._switch_tab(2)
        try:
            self._draw_traj_static()
            self._update_traj(0)
        except Exception:
            pass
        extra = ""
        if metodo == "ctrl_cinematico" and err_ee_full is not None:
            err_max = np.max(np.linalg.norm(err_ee_full, axis=1))
            extra = f"  |e|max={err_max:.3f}mm  K={K_gain:.1f}"
        self._lbl_status.config(
            text=f"Ping pong [{metodo}] cargado en pestaña 3 — "
                 f"{len(keyframes_q)} waypoints, "
                 f"tf={t_arr[-1]:.2f}s.{extra}  Presiona ▶ para animar.")

    # ═══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA 6 — ROS2 / GAZEBO
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_tab_ros2(self, parent):
        # ── Cabecera ──
        hdr = tk.Frame(parent, bg=PANEL, pady=10)
        hdr.pack(fill="x", side="top")
        tk.Frame(hdr, bg=BORDER, height=1).pack(fill="x")
        row = tk.Frame(hdr, bg=PANEL)
        row.pack(fill="x", padx=14, pady=6)
        tk.Label(row, text="BRIDGE ROS2 / GAZEBO",
                 bg=PANEL, fg=ACCENT,
                 font=("Courier New", 12, "bold")).pack(side="left")
        tk.Label(row,
                 text="  Publica Float64MultiArray a /joint_group_position_controller/commands",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=8)
        tk.Frame(hdr, bg=BORDER, height=1).pack(fill="x")

        if not ROS2_OK:
            tk.Label(parent,
                     text="\n\n  rclpy no encontrado.\n\n"
                          "  Para activar esta pestaña instala ROS2 Jazzy y "
                          "ejecuta la app desde un terminal con ROS2 configurado:\n\n"
                          "    source /opt/ros/jazzy/setup.bash\n"
                          "    python3 'brazo_trayectoria_3dof (8).py'",
                     bg=BG, fg="#f85149",
                     font=("Courier New", 11), justify="left").pack(anchor="nw", padx=20)
            return

        body = tk.Frame(parent, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=10)

        # ── Panel de conexion ──
        conn = tk.Frame(body, bg=PANEL, pady=8)
        conn.pack(fill="x")

        self._btn_ros2_conn = tk.Button(
            conn, text="  Conectar ROS2  ",
            bg=BTN_PLAY, fg=ACCENT, relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=14, pady=6,
            command=self._ros2_toggle)
        self._btn_ros2_conn.pack(side="left", padx=14)

        self._lbl_ros2_status = tk.Label(
            conn, text="desconectado", bg=PANEL, fg="#f85149",
            font=("Courier New", 10, "bold"))
        self._lbl_ros2_status.pack(side="left", padx=8)

        tk.Frame(conn, bg=BORDER, width=2, height=30).pack(side="left", padx=14)

        # Auto-publish toggle
        self._ros2_auto_pub_var = tk.BooleanVar(value=False)
        ck = tk.Checkbutton(
            conn,
            text="  Auto-publicar al mover sliders (pestaña 1)",
            variable=self._ros2_auto_pub_var,
            bg=PANEL, fg=TEXT, selectcolor=BTN_HOV,
            activebackground=PANEL, activeforeground=ACCENT,
            font=("Courier New", 10), cursor="hand2",
            command=self._ros2_on_auto_toggle)
        ck.pack(side="left", padx=4)

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(10, 0))

        # ── Sección: envio manual ──
        tk.Label(body, text="ENVIAR POSICIÓN ACTUAL AL GAZEBO",
                 bg=BG, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(anchor="w", pady=(10, 4))

        send_frm = tk.Frame(body, bg=BG)
        send_frm.pack(fill="x")

        tk.Button(send_frm,
                  text="  ► Enviar q_actual (Vista 3D → Gazebo)  ",
                  bg=BTN_HOV, fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=14, pady=7,
                  command=lambda: self._ros2_publish_q(self._q_current)
                  ).pack(side="left")

        tk.Button(send_frm,
                  text="  ⊙ Enviar cero (0, 0, 0)  ",
                  bg=BTN_BG, fg=TEXT, relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=14, pady=7,
                  command=lambda: self._ros2_publish_q(np.zeros(3))
                  ).pack(side="left", padx=10)

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(10, 0))

        # ── Sección: feedback de Gazebo ──
        tk.Label(body, text="RETROALIMENTACIÓN DE GAZEBO  (/joint_states)",
                 bg=BG, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(anchor="w", pady=(10, 4))

        fb_frm = tk.Frame(body, bg="#0a0e14",
                          highlightbackground=BORDER, highlightthickness=1)
        fb_frm.pack(fill="x", ipady=6)

        row = tk.Frame(fb_frm, bg="#0a0e14")
        row.pack(fill="x", padx=12, pady=4)
        self._ros2_feedback_labels = []
        joint_names_urdf = ["J1", "J2", "J3"]
        for j in range(3):
            tk.Label(row, text=f"{joint_names_urdf[j]}=",
                     bg="#0a0e14", fg=C_J[j],
                     font=("Courier New", 10)).pack(side="left", padx=(14, 0))
            lbl = tk.Label(row, text="  ---  rad", width=14, anchor="w",
                           bg="#0a0e14", fg=ACCENT,
                           font=("Courier New", 10, "bold"))
            lbl.pack(side="left")
            self._ros2_feedback_labels.append(lbl)

        row2 = tk.Frame(fb_frm, bg="#0a0e14")
        row2.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(row2,
                 text="(en grados)  →",
                 bg="#0a0e14", fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=14)
        self._ros2_feedback_deg_labels = []
        for j in range(3):
            tk.Label(row2, text=f"{joint_names_urdf[j]}=",
                     bg="#0a0e14", fg=C_J[j],
                     font=("Courier New", 9)).pack(side="left", padx=(14, 0))
            lbl = tk.Label(row2, text="  ---  °", width=12, anchor="w",
                           bg="#0a0e14", fg=MUTED,
                           font=("Courier New", 9))
            lbl.pack(side="left")
            self._ros2_feedback_deg_labels.append(lbl)

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(10, 0))

        # ── Sección: streaming de trayectoria ──
        tk.Label(body, text="STREAMING DE TRAYECTORIA A GAZEBO",
                 bg=BG, fg=ACCENT,
                 font=("Courier New", 10, "bold")).pack(anchor="w", pady=(10, 4))
        tk.Label(body,
                 text="  Planifica la trayectoria en la pestaña 3 primero. "
                      "No requiere conexión Serial.",
                 bg=BG, fg=MUTED,
                 font=("Courier New", 9), justify="left").pack(anchor="w", padx=14)

        str_frm = tk.Frame(body, bg=BG)
        str_frm.pack(fill="x", pady=(6, 0))

        tk.Label(str_frm, text="Frecuencia (Hz):", bg=BG, fg=MUTED,
                 font=("Courier New", 9)).pack(side="left", padx=(0, 4))
        self._ros2_stream_hz = tk.StringVar(value="20")
        tk.Entry(str_frm, textvariable=self._ros2_stream_hz, width=5,
                 bg="#0a0e14", fg=ACCENT, insertbackground=ACCENT,
                 relief="flat", font=("Courier New", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=BTN_HOV).pack(side="left")

        self._btn_ros2_stream = tk.Button(
            str_frm, text="  ▶ Enviar trayectoria a Gazebo  ",
            bg=BTN_PLAY, fg=ACCENT, relief="flat", bd=0,
            font=("Courier New", 10, "bold"), cursor="hand2",
            padx=12, pady=5,
            command=self._ros2_start_stream)
        self._btn_ros2_stream.pack(side="left", padx=10)

        tk.Button(str_frm, text="  ■ Detener  ",
                  bg="#b62324", fg=ACCENT, relief="flat", bd=0,
                  font=("Courier New", 10, "bold"), cursor="hand2",
                  padx=10, pady=5,
                  command=self._ros2_stop_stream).pack(side="left")

        self._lbl_ros2_stream = tk.Label(body, text="", bg=BG, fg=MUTED,
                                         font=("Courier New", 9))
        self._lbl_ros2_stream.pack(anchor="w", padx=4)

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(10, 0))

        # ── Sección: info ──
        info = (
            "  Mapeo de articulaciones:  GUI q1→J1, q2→J2, q3→J3\n"
            "  Conversión de unidades:   grados → radianes  (×π/180), J2 y J3 negados\n"
            "  Tópico publicado:         /joint_group_position_controller/commands\n"
            "  Tipo de mensaje:          std_msgs/Float64MultiArray  [J1, J2, J3]\n"
            "  Tópico suscrito:          /joint_states  (feedback de posición)\n"
            "  Nota: al usar Serial+ROS2 simultáneamente, la trayectoria se envía a ambos."
        )
        tk.Label(body, text=info,
                 bg=BG, fg=MUTED,
                 font=("Courier New", 9), justify="left").pack(anchor="w", pady=(10, 0))

    # ──────────────────────────────────────────
    #  Streaming de trayectoria a Gazebo
    # ──────────────────────────────────────────
    def _ros2_start_stream(self):
        if not self._ros2_running:
            messagebox.showwarning("ROS2 no conectado",
                                   "Conecta ROS2 primero (botón 'Conectar ROS2').")
            return
        if self._tr_data is None:
            messagebox.showwarning("Sin trayectoria",
                                   "Planifica una trayectoria en la pestaña 3 primero.")
            return
        if self._ros2_stream_thread is not None and self._ros2_stream_thread.is_alive():
            messagebox.showwarning("Stream en curso",
                                   "Ya hay un stream activo. Deténlo primero.")
            return
        try:
            hz = float(self._ros2_stream_hz.get())
            if hz <= 0 or hz > 200:
                raise ValueError
        except ValueError:
            messagebox.showerror("Frecuencia inválida", "Usa un valor entre 1 y 200 Hz.")
            return
        self._ros2_stream_stop.clear()
        self._ros2_stream_thread = threading.Thread(
            target=self._ros2_stream_worker, args=(hz,), daemon=True)
        self._ros2_stream_thread.start()
        try:
            self._lbl_ros2_stream.config(
                text=f"  Iniciando stream a {hz:.0f} Hz…", fg="#3fb950")
        except Exception:
            pass

    def _ros2_stop_stream(self):
        self._ros2_stream_stop.set()
        try:
            self._lbl_ros2_stream.config(text="  Stream detenido.", fg=MUTED)
        except Exception:
            pass

    def _ros2_stream_worker(self, hz):
        """Envía la trayectoria planificada a Gazebo punto a punto.
        Si hay grip_events en _tr_data, los dispara via serial (si está conectado)."""
        d = self._tr_data
        if d is None:
            return
        q  = d["q"]     # (3, N) en grados
        tt = d["t"]     # (N,)
        tf = d["tf"]
        N  = len(tt)
        dt = tt[1] - tt[0] if N > 1 else 0.01

        period  = 1.0 / hz
        t_start = time.monotonic()
        sent    = 0

        grip_events = sorted(d.get("grip_events") or [], key=lambda e: e[0])
        g_idx = 0

        try:
            # Disparar eventos en t=0
            while g_idx < len(grip_events) and grip_events[g_idx][0] <= 0.0:
                self._send_json_threadsafe({"grip": grip_events[g_idx][1]})
                g_idx += 1

            while not self._ros2_stream_stop.is_set():
                elapsed = time.monotonic() - t_start
                # Disparar grip events cuyo tiempo ha llegado
                while g_idx < len(grip_events) and elapsed >= grip_events[g_idx][0]:
                    self._send_json_threadsafe({"grip": grip_events[g_idx][1]})
                    g_idx += 1
                if elapsed >= tf:
                    self._ros2_publish_q(q[:, -1])
                    while g_idx < len(grip_events):
                        self._send_json_threadsafe({"grip": grip_events[g_idx][1]})
                        g_idx += 1
                    break
                i = min(int(elapsed / dt), N - 1)
                self._ros2_publish_q(q[:, i])
                sent += 1

                try:
                    self._lbl_ros2_stream.after(
                        0, lambda s=sent, el=elapsed:
                        self._lbl_ros2_stream.config(
                            text=f"  enviados: {s}  |  t={el:.2f}s / {tf:.1f}s",
                            fg="#3fb950"))
                except Exception:
                    pass

                next_tick = t_start + sent * period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    if self._ros2_stream_stop.wait(sleep_s):
                        break
        except Exception as e:
            try:
                self.after(0, lambda: self._lbl_ros2_stream.config(
                    text=f"  Error: {e}", fg="#f85149"))
            except Exception:
                pass

        try:
            self.after(0, lambda s=sent: self._lbl_ros2_stream.config(
                text=f"  Stream finalizado — {s} puntos enviados.", fg=MUTED))
        except Exception:
            pass

    def _ros2_toggle(self):
        if self._ros2_running:
            self._ros2_disconnect()
        else:
            self._ros2_connect()

    def _ros2_on_auto_toggle(self):
        self._ros2_auto_pub = self._ros2_auto_pub_var.get()

    def _ros2_connect(self):
        if not ROS2_OK:
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._ros2_node = rclpy.create_node('brazo_gui_bridge')
            self._ros2_pub = self._ros2_node.create_publisher(
                Float64MultiArray,
                '/joint_group_position_controller/commands',
                10)
            self._ros2_node.create_subscription(
                JointState,
                '/joint_states',
                self._ros2_joint_state_cb,
                10)
            self._ros2_running = True
            self._ros2_thread = threading.Thread(
                target=self._ros2_spin_worker, daemon=True)
            self._ros2_thread.start()
            try:
                self._lbl_ros2_status.config(
                    text="conectado — nodo: brazo_gui_bridge", fg="#3fb950")
                self._btn_ros2_conn.config(
                    text="  Desconectar ROS2  ", bg=BTN_PAUS)
            except Exception:
                pass
            self._lbl_status.config(text="ROS2: bridge iniciado.")
        except Exception as e:
            try:
                self._lbl_ros2_status.config(text=f"error: {e}", fg="#f85149")
            except Exception:
                pass

    def _ros2_disconnect(self):
        self._ros2_running = False
        self._ros2_auto_pub = False
        self._ros2_last_cmd = None
        try:
            self._ros2_auto_pub_var.set(False)
        except Exception:
            pass
        if self._ros2_thread is not None:
            self._ros2_thread.join(timeout=1.0)
            self._ros2_thread = None
        if self._ros2_node is not None:
            try:
                self._ros2_node.destroy_node()
            except Exception:
                pass
            self._ros2_node = None
        self._ros2_pub = None
        try:
            self._lbl_ros2_status.config(text="desconectado", fg="#f85149")
            self._btn_ros2_conn.config(
                text="  Conectar ROS2  ", bg=BTN_PLAY)
        except Exception:
            pass

    def _ros2_spin_worker(self):
        # Republica el ultimo comando a ~20 Hz para mantener el controlador activo.
        # JointGroupPositionController necesita un stream continuo; sin esto
        # el primer comando se pierde o no se ejecuta hasta que llega el siguiente.
        PUB_INTERVAL = 0.05   # 20 Hz
        last_pub = 0.0
        try:
            while self._ros2_running and rclpy.ok():
                rclpy.spin_once(self._ros2_node, timeout_sec=0.01)
                now = time.monotonic()
                if now - last_pub >= PUB_INTERVAL and self._ros2_last_cmd is not None:
                    try:
                        self._ros2_pub.publish(self._ros2_last_cmd)
                    except Exception:
                        pass
                    last_pub = now
        except Exception:
            pass

    def _ros2_publish_q(self, q_deg):
        """Publica posiciones articulares (grados) al controlador de Gazebo.
        Almacena el mensaje como ultimo comando para que el hilo spin lo
        repubique a 20 Hz y el controlador no pierda la referencia.

        Convencion de signos (SolidWorks vs GUI):
          J1: directo  — limites continuos, sin restriccion de signo
          J2: negado   — URDF J2 es continuous; SolidWorks define positivo
                         en sentido opuesto al encoder fisico, por lo que se
                         invierte el signo para que el movimiento en Gazebo
                         corresponda con el del robot real.
          J3: negado   — Igual que J2: SolidWorks invierte la convencion de
                         signo respecto al encoder fisico.
        """
        if not self._ros2_running or self._ros2_pub is None:
            return
        try:
            msg = Float64MultiArray()
            signs = [1.0, -1.0, -1.0]  # J2 y J3 invertidos por convencion de ejes URDF
            msg.data = [float(signs[i] * np.deg2rad(float(q_deg[i]))) for i in range(3)]
            self._ros2_last_cmd = msg
            self._ros2_pub.publish(msg)
        except Exception:
            pass

    def _ros2_joint_state_cb(self, msg):
        """Callback del suscriptor /joint_states — actualiza feedback de Gazebo."""
        joint_map = {'J1': 0, 'J2': 1, 'J3': 2}
        for name, pos in zip(msg.name, msg.position):
            if name in joint_map:
                self._ros2_feedback_q[joint_map[name]] = float(pos)
        try:
            self.after(0, self._ros2_update_feedback_ui)
        except Exception:
            pass

    def _ros2_update_feedback_ui(self):
        """Actualiza las etiquetas de retroalimentación en la UI (hilo principal)."""
        for j in range(3):
            rad = self._ros2_feedback_q[j]
            deg = np.rad2deg(rad)
            try:
                self._ros2_feedback_labels[j].config(
                    text=f"{rad:+8.4f} rad")
                self._ros2_feedback_deg_labels[j].config(
                    text=f"{deg:+8.2f}°")
            except (IndexError, tk.TclError):
                pass

    # ──────────────────────────────────────────────────────────
    def _on_close(self):
        if self._job is not None:
            try: self.after_cancel(self._job)
            except Exception: pass
        # Parar streaming y cerrar serial
        try:
            self._stream_stop.set()
            self._disconnect_serial()
        except Exception:
            pass
        # Cerrar ROS2
        try:
            self._ros2_stop_stream()
            self._ros2_disconnect()
        except Exception:
            pass
        plt.close("all")
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    BrazoApp().mainloop()
