import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import messagebox
import threading

# ── ROS2 bridge (optional — works with or without ROS2) ──────────────
try:
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration

    class GazeboBridge(Node):
        def __init__(self):
            super().__init__('gui_gazebo_bridge')
            self.publisher = self.create_publisher(
                JointTrajectory,
                '/joint_trajectory_controller/joint_trajectory',
                10
            )

        def send_joints(self, r1_rad, r2_rad, r3_rad, duration_sec=0.25):
            # Subtract offsets so GUI 0° = physical center (Gazebo 0 = offset position)
            r1_rad -= OFFSET_R0
            r2_rad -= OFFSET_R1
            r3_rad -= OFFSET_R2
            traj = JointTrajectory()
            traj.joint_names = ['R1', 'R2', 'R3']
            # Send multiple waypoints for smooth motion
            steps = 20
            for i in range(1, steps + 1):
                frac = i / steps
                point = JointTrajectoryPoint()
                point.positions = [r1_rad, r2_rad, r3_rad]
                ns = int((duration_sec * frac) * 1e9)
                point.time_from_start = Duration(sec=int(duration_sec * frac), nanosec=ns % 1_000_000_000)
                traj.points.append(point)
            self.publisher.publish(traj)

    rclpy.init()
    _bridge = GazeboBridge()
    _ros_thread = threading.Thread(target=rclpy.spin, args=(_bridge,), daemon=True)
    _ros_thread.start()
    ROS_ENABLED = True
    print("[ROS2] Gazebo bridge active — sliders will control the robot!")
except Exception as e:
    ROS_ENABLED = False
    print(f"[ROS2] Not available ({e}) — running in visualization-only mode.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║               ROBOT CONFIGURATION  — EDIT HERE                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Link lengths (any positive number, same units)                 ║
A1 = 18.0          # Length of link 1 (shoulder → elbow)
A2 = 14.0          # Length of link 2 (elbow → end-effector)
#                                                                    ║
# ║  Joint angle limits  (degrees)                                  ║
T0_MIN, T0_MAX = -180, 180   # Base rotation  (around vertical Z)
T1_MIN, T1_MAX = -180, 270   # Shoulder pitch (negative = below base plane)
T2_MIN, T2_MAX =  -180, 270   # Elbow          (0 = straight, 180 = fully folded)
#                     
#                                                ║
Ang_rad0=0
Ang_rad1=0
Ang_rad2=0

# Joint zero offsets — added to every command sent to Gazebo
OFFSET_R0 = 1.5638
OFFSET_R1 = -0.4702
OFFSET_R2 = -1.5534
# ║  Default starting angles when opening the visualizer (degrees)  ║
T0_DEFAULT =   Ang_rad0*180/3.141516
T1_DEFAULT =  Ang_rad1*180/3.141516
T2_DEFAULT =  Ang_rad2*180/3.141516
# ╚══════════════════════════════════════════════════════════════════╝

# Derived — do not edit below this line
LIM = A1 + A2 + 2

# ─────────────────────────────────────────────
#  Kinematics
# ─────────────────────────────────────────────
def dh_matrix(a, d, alpha, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])

def forward_kinematics(t0, t1, t2):
    """All angles in radians. Returns joint positions + cumulative DH matrices."""
    T0 = dh_matrix(0,   0, 0, t0)
    T1 = dh_matrix(A1,  0, 0, t1)
    T2 = dh_matrix(A2,  0, 0, t2)

    r1 = A1 * math.cos(t1)
    z1 = A1 * math.sin(t1)
    r2 = r1 + A2 * math.cos(t1 + t2)
    z2 = z1 + A2 * math.sin(t1 + t2)

    def to3d(r, z):
        return r * math.cos(t0), r * math.sin(t0), z

    p0 = (0.0, 0.0, 0.0)
    p1 = to3d(r1, z1)
    p2 = to3d(r2, z2)

    T01   = T0
    T012  = T0 @ T1
    T0123 = T0 @ T1 @ T2
    return p0, p1, p2, T01, T012, T0123

def inverse_kinematics(x, y, z):
    """Returns (t0, t1, t2) in degrees."""
    t0 = math.atan2(y, x)
    r  = math.sqrt(x**2 + y**2)
    dist = math.sqrt(r**2 + z**2)
    if dist > A1 + A2 or dist < abs(A1 - A2):
        raise ValueError(f"Point outside workspace (max reach = {A1+A2:.1f}).")
    c2 = (r**2 + z**2 - A1**2 - A2**2) / (2 * A1 * A2)
    c2 = max(-1.0, min(1.0, c2))
    t2 = math.acos(c2)
    k1 = A1 + A2 * math.cos(t2)
    k2 = A2 * math.sin(t2)
    t1 = math.atan2(z, r) - math.atan2(k2, k1)
    return math.degrees(t0), math.degrees(t1), math.degrees(t2)

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def matrix_str(M, label):
    lines = [f"  {label}"]
    for row in M:
        lines.append("  [" + "  ".join(f"{v:7.3f}" for v in row) + " ]")
    return "\n".join(lines)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ─────────────────────────────────────────────
#  Interactive visualizer
# ─────────────────────────────────────────────
def open_interactive(init_t0=T0_DEFAULT, init_t1=T1_DEFAULT, init_t2=T2_DEFAULT):
    init_t0 = clamp(init_t0, T0_MIN, T0_MAX)
    init_t1 = clamp(init_t1, T1_MIN, T1_MAX)
    init_t2 = clamp(init_t2, T2_MIN, T2_MAX)

    fig = plt.figure(figsize=(15, 8), facecolor="#1e1e2e")
    fig.suptitle(
        f"3-DOF Robot Arm   |   a₁={A1}  a₂={A2}   |   "
        f"θ₀ [{T0_MIN}°,{T0_MAX}°]  "
        f"θ₁ [{T1_MIN}°,{T1_MAX}°]  "
        f"θ₂ [{T2_MIN}°,{T2_MAX}°]",
        color="#a6adc8", fontsize=9
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[2.2, 1.2, 1.0],
        height_ratios=[3, 1],
        hspace=0.05, wspace=0.35,
        left=0.05, right=0.97, top=0.93, bottom=0.06
    )

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.set_facecolor("#13131f")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#444")

    ax_mat = fig.add_subplot(gs[0, 1])
    ax_mat.set_facecolor("#13131f")
    ax_mat.axis("off")
    mat_text = ax_mat.text(
        0.02, 0.98, "", transform=ax_mat.transAxes,
        fontsize=7.2, color="#a9e34b", va="top", ha="left", fontfamily="monospace"
    )
    ax_mat.set_title("Homogeneous Matrices", color="#cdd6f4", fontsize=9, pad=4)

    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.set_facecolor("#13131f")
    ax_info.axis("off")
    info_text = ax_info.text(
        0.05, 0.95, "", transform=ax_info.transAxes,
        fontsize=9, color="#cdd6f4", va="top", ha="left", fontfamily="monospace"
    )
    ax_info.set_title("Joint / Position Info", color="#cdd6f4", fontsize=9, pad=4)

    fig.add_subplot(gs[1, 0]).axis("off")

    sl_col = "#313244"
    sl_ax0 = fig.add_axes([0.06, 0.075, 0.38, 0.022], facecolor=sl_col)
    sl_ax1 = fig.add_axes([0.06, 0.045, 0.38, 0.022], facecolor=sl_col)
    sl_ax2 = fig.add_axes([0.06, 0.015, 0.38, 0.022], facecolor=sl_col)
    btn_ax = fig.add_axes([0.455, 0.010, 0.08, 0.065], facecolor="#313244")

    s0 = Slider(sl_ax0, f"θ₀ Base  ({T0_MIN}° – {T0_MAX}°)",
                T0_MIN, T0_MAX, valinit=init_t0, color="#89b4fa")
    s1 = Slider(sl_ax1, f"θ₁ Shoulder  ({T1_MIN}° – {T1_MAX}°)",
                T1_MIN, T1_MAX, valinit=init_t1, color="#f9e2af")
    s2 = Slider(sl_ax2, f"θ₂ Elbow  ({T2_MIN}° – {T2_MAX}°)",
                T2_MIN, T2_MAX, valinit=init_t2, color="#a6e3a1")

    for sl in (s0, s1, s2):
        sl.label.set_color("white")
        sl.valtext.set_color("#cdd6f4")

    def draw(t0_deg, t1_deg, t2_deg):
        t0r, t1r, t2r = map(math.radians, [t0_deg, t1_deg, t2_deg])
        p0, p1, p2, T01, T012, T0123 = forward_kinematics(t0r, t1r, t2r)

        ax3d.cla()
        ax3d.set_facecolor("#13131f")
        ax3d.set_xlim(-LIM, LIM)
        ax3d.set_ylim(-LIM, LIM)
        ax3d.set_zlim(-LIM, LIM)
        ax3d.set_xlabel("X", color="#cdd6f4")
        ax3d.set_ylabel("Y", color="#cdd6f4")
        ax3d.set_zlabel("Z", color="#cdd6f4")
        ax3d.tick_params(colors="#585b70")
        ax3d.set_title(
            f"End-effector:  x={p2[0]:.2f}  y={p2[1]:.2f}  z={p2[2]:.2f}",
            color="#89dceb", fontsize=9
        )

        # Workspace sphere
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        R = A1 + A2
        ax3d.plot_wireframe(
            R*np.outer(np.cos(u), np.sin(v)),
            R*np.outer(np.sin(u), np.sin(v)),
            R*np.outer(np.ones_like(u), np.cos(v)),
            color="#313244", linewidth=0.3, alpha=0.25
        )

        # Subtle ground grid
        gv = np.linspace(-LIM, LIM, 9)
        for g in gv:
            ax3d.plot([g, g], [-LIM, LIM], [-LIM, -LIM], color="#2a2a3e", lw=0.5, alpha=0.5)
            ax3d.plot([-LIM, LIM], [g, g], [-LIM, -LIM], color="#2a2a3e", lw=0.5, alpha=0.5)

        # Links
        ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                  color="#89b4fa", lw=5, solid_capstyle="round")
        ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                  color="#a6e3a1", lw=5, solid_capstyle="round")

        # Joints
        for pt, col, lbl in [(p0, "#f38ba8", "Base"),
                              (p1, "#89b4fa", "Elbow"),
                              (p2, "#f9e2af", "EE")]:
            ax3d.scatter(*pt, color=col, s=90, zorder=5, depthshade=False)
            ax3d.text(pt[0], pt[1], pt[2] + 0.8, lbl, color=col, fontsize=7)

        # Ground shadow
        ax3d.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]],
                  [-LIM, -LIM, -LIM],
                  color="#45475a", lw=1.5, linestyle="--", alpha=0.5)

        mat_text.set_text(
            matrix_str(T01,   "T₀→₁  (base rotation)") + "\n\n" +
            matrix_str(T012,  "T₀→₂  (+ shoulder)") + "\n\n" +
            matrix_str(T0123, "T₀→₃  (end-effector)")
        )

        info_text.set_text(
            f"  Joint angles\n"
            f"  ─────────────────\n"
            f"  θ₀  base       : {t0_deg:8.2f}°\n"
            f"  θ₁  shoulder   : {t1_deg:8.2f}°\n"
            f"  θ₂  elbow      : {t2_deg:8.2f}°\n\n"
            f"  End-Effector\n"
            f"  ─────────────────\n"
            f"  x  : {p2[0]:8.3f}\n"
            f"  y  : {p2[1]:8.3f}\n"
            f"  z  : {p2[2]:8.3f}\n\n"
            f"  Config\n"
            f"  ─────────────────\n"
            f"  a₁ : {A1}\n"
            f"  a₂ : {A2}\n"
            f"  max reach : {A1+A2:.1f}"
        )

        fig.canvas.draw_idle()

    from matplotlib.widgets import Button as MplButton
    import time

    _target = [0.0, 0.0, 0.0]       # desired joint positions (radians)
    _last_slider_time = [0.0]
    _tracker_running = [False]

    def _tracking_loop():
        """Keep resending the target every 0.5s until slider moves again."""
        import time
        while True:
            time.sleep(0.05)
            # If slider was moved recently, skip — on_slider already sent it
            if time.time() - _last_slider_time[0] < 0.4:
                continue
            if ROS_ENABLED:
                _bridge.send_joints(_target[0], _target[1], _target[2], duration_sec=0.25)

    # Start tracking loop once
    _tracker_thread = threading.Thread(target=_tracking_loop, daemon=True)
    _tracker_thread.start()

    def on_slider(_):
        draw(s0.val, s1.val, s2.val)
        _target[0] = math.radians(s0.val)
        _target[1] = math.radians(s1.val)
        _target[2] = math.radians(s2.val)
        _last_slider_time[0] = time.time()
        if ROS_ENABLED:
            _bridge.send_joints(_target[0], _target[1], _target[2], duration_sec=0.25)

    def center_robot(_):
        s0.set_val(0.0)
        s1.set_val(0.0)
        s2.set_val(0.0)
        _target[0] = 0.0
        _target[1] = 0.0
        _target[2] = 0.0
        if ROS_ENABLED:
            _bridge.send_joints(0.0, 0.0, 0.0, duration_sec=0.25)

    btn_center = MplButton(btn_ax, 'Center\nRobot', color='#45475a', hovercolor='#585b70')
    btn_center.label.set_color('#cdd6f4')
    btn_center.label.set_fontsize(8)
    btn_center.on_clicked(center_robot)

    s0.on_changed(on_slider)
    s1.on_changed(on_slider)
    s2.on_changed(on_slider)

    draw(init_t0, init_t1, init_t2)
    plt.show()


# ─────────────────────────────────────────────
#  Pop-up menu
# ─────────────────────────────────────────────
def popup_menu():
    root = tk.Tk()
    root.withdraw()

    while True:
        dialog = tk.Toplevel(root)
        dialog.title("3-DOF Robot Arm")
        dialog.resizable(False, False)
        dialog.configure(bg="#1e1e2e")
        dialog.grab_set()

        tk.Label(dialog, text="3-DOF Robot Arm Kinematics",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Helvetica", 14, "bold"), pady=10).pack()
        tk.Label(dialog,
                 text=f"a₁ = {A1}   a₂ = {A2}   |   max reach = {A1+A2:.1f}",
                 bg="#1e1e2e", fg="#a6adc8", font=("Helvetica", 9)).pack(pady=(0, 4))
        tk.Label(dialog,
                 text=f"θ₀ [{T0_MIN}°,{T0_MAX}°]   "
                      f"θ₁ [{T1_MIN}°,{T1_MAX}°]   "
                      f"θ₂ [{T2_MIN}°,{T2_MAX}°]",
                 bg="#1e1e2e", fg="#585b70", font=("Helvetica", 8)).pack(pady=(0, 10))

        choice = tk.StringVar(value="")
        btn = dict(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                   activeforeground="white", relief="flat", padx=14, pady=8,
                   font=("Helvetica", 11), width=30, cursor="hand2")

        def pick(val):
            choice.set(val)
            dialog.destroy()

        tk.Button(dialog, text="▶  Forward Kinematics  (angles → position)",
                  command=lambda: pick("forward"), **btn).pack(pady=4, padx=20)
        tk.Button(dialog, text="◀  Inverse Kinematics  (position → angles)",
                  command=lambda: pick("inverse"), **btn).pack(pady=4, padx=20)
        tk.Frame(dialog, bg="#313244", height=1).pack(fill="x", padx=20, pady=8)
        tk.Button(dialog, text="✕  Exit",
                  command=lambda: pick("exit"),
                  bg="#313244", fg="#f38ba8", activebackground="#45475a",
                  activeforeground="#f38ba8", relief="flat", padx=14, pady=8,
                  font=("Helvetica", 11), width=30, cursor="hand2").pack(pady=(0, 14), padx=20)

        dialog.wait_window()
        c = choice.get()

        if c in ("exit", ""):
            break

        # ── Forward kinematics ────────────────────────────────────────
        elif c == "forward":
            sub = tk.Toplevel(root)
            sub.title("Forward Kinematics — Initial Angles")
            sub.configure(bg="#1e1e2e")
            sub.resizable(False, False)
            sub.grab_set()

            tk.Label(sub, text="Set initial joint angles",
                     bg="#1e1e2e", fg="#cdd6f4",
                     font=("Helvetica", 12, "bold"), pady=10).pack()

            fields = [
                (f"θ₀  base  ({T0_MIN}° to {T0_MAX}°)",      str(T0_DEFAULT)),
                (f"θ₁  shoulder  ({T1_MIN}° to {T1_MAX}°)",  str(T1_DEFAULT)),
                (f"θ₂  elbow  ({T2_MIN}° to {T2_MAX}°)",     str(T2_DEFAULT)),
            ]
            entries_fk = []
            for label, default in fields:
                row = tk.Frame(sub, bg="#1e1e2e")
                row.pack(fill="x", padx=20, pady=3)
                tk.Label(row, text=label, bg="#1e1e2e", fg="#a6adc8",
                         font=("Helvetica", 10), width=30, anchor="w").pack(side="left")
                e = tk.Entry(row, bg="#313244", fg="#cdd6f4", insertbackground="white",
                             font=("Helvetica", 10), width=8, relief="flat")
                e.insert(0, default)
                e.pack(side="left", padx=6)
                entries_fk.append(e)

            fk_result = []

            def confirm_fk():
                fk_result.extend([e.get() for e in entries_fk])
                sub.destroy()

            tk.Button(sub, text="Open Visualizer", command=confirm_fk,
                      bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec",
                      relief="flat", pady=7, font=("Helvetica", 10, "bold"),
                      width=18, cursor="hand2").pack(pady=(10, 4))
            tk.Button(sub, text="Cancel", command=sub.destroy,
                      bg="#313244", fg="#cdd6f4", relief="flat", pady=7,
                      font=("Helvetica", 10), width=18, cursor="hand2").pack(pady=(0, 12))

            sub.wait_window()

            if fk_result:
                try:
                    t0, t1, t2 = float(fk_result[0]), float(fk_result[1]), float(fk_result[2])
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers.")
                    continue
                open_interactive(t0, t1, t2)

        # ── Inverse kinematics ────────────────────────────────────────
        elif c == "inverse":
            sub = tk.Toplevel(root)
            sub.title("Inverse Kinematics — Target Position")
            sub.configure(bg="#1e1e2e")
            sub.resizable(False, False)
            sub.grab_set()

            tk.Label(sub, text="Enter target end-effector position",
                     bg="#1e1e2e", fg="#cdd6f4",
                     font=("Helvetica", 12, "bold"), pady=10).pack()
            tk.Label(sub, text=f"Max reach: {A1+A2:.1f}   (a₁={A1}, a₂={A2})",
                     bg="#1e1e2e", fg="#a6adc8", font=("Helvetica", 9)).pack(pady=(0, 8))

            ik_fields = [("x", "10"), ("y", "5"), ("z", "8")]
            entries_ik = []
            for label, default in ik_fields:
                row = tk.Frame(sub, bg="#1e1e2e")
                row.pack(fill="x", padx=20, pady=3)
                tk.Label(row, text=label, bg="#1e1e2e", fg="#a6adc8",
                         font=("Helvetica", 10), width=6, anchor="w").pack(side="left")
                e = tk.Entry(row, bg="#313244", fg="#cdd6f4", insertbackground="white",
                             font=("Helvetica", 10), width=10, relief="flat")
                e.insert(0, default)
                e.pack(side="left", padx=6)
                entries_ik.append(e)

            ik_result = []

            def confirm_ik():
                ik_result.extend([e.get() for e in entries_ik])
                sub.destroy()

            tk.Button(sub, text="Solve & Open Visualizer", command=confirm_ik,
                      bg="#a6e3a1", fg="#1e1e2e", activebackground="#94e2d5",
                      relief="flat", pady=7, font=("Helvetica", 10, "bold"),
                      width=22, cursor="hand2").pack(pady=(10, 4))
            tk.Button(sub, text="Cancel", command=sub.destroy,
                      bg="#313244", fg="#cdd6f4", relief="flat", pady=7,
                      font=("Helvetica", 10), width=22, cursor="hand2").pack(pady=(0, 12))

            sub.wait_window()

            if ik_result:
                try:
                    x, y, z = float(ik_result[0]), float(ik_result[1]), float(ik_result[2])
                    t0, t1, t2 = inverse_kinematics(x, y, z)
                except ValueError as err:
                    messagebox.showerror("Error", str(err))
                    continue
                open_interactive(t0, t1, t2)

    root.destroy()


if __name__ == "__main__":
    popup_menu()
