#!/bin/bash
# Launches the brazo ROS2 project:
#   Window 1 — Gazebo simulation (gz_simulator_launch.py)
#   Window 2 — Trajectory GUI (brazo_trayectoria_3dof)

ROS_SETUP="/opt/ros/jazzy/setup.bash"
WS_SETUP="/home/arlo/Ros_projects/Ros2/ass12/install/setup.bash"
GUI="/home/arlo/Ros_projects/Ros2/brazo_trayectoria_3dof (8).py"

# Verify the workspace has been built
if [ ! -f "$WS_SETUP" ]; then
    echo "[ERROR] Workspace not built — run 'colcon build' inside ass12/ first."
    exit 1
fi

echo "[brazo] Starting Gazebo simulation..."
gnome-terminal --title="Brazo — Gazebo" -- bash -c "
    source '$ROS_SETUP'
    source '$WS_SETUP'
    ros2 launch asse12 gz_simulator_launch.py
    exec bash
"

echo "[brazo] Starting GUI (2 s delay to let Gazebo init)..."
sleep 2
gnome-terminal --title="Brazo — GUI" -- bash -c "
    source '$ROS_SETUP'
    source '$WS_SETUP'
    python3 '$GUI'
    exec bash
"

echo "[brazo] Done — two windows opened."
