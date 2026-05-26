import os
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import launch_ros.descriptions
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from pathlib import Path
from launch.conditions import IfCondition

def generate_launch_description():
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    urdf_file= LaunchConfiguration('urdf_file')
    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration('use_sim_time')
    bringup_dir = get_package_share_directory('asse12')
    world = '/home/arlo/Ros_projects/Ros2/brazo_workspace.sdf'
    urdf_file_path = os.path.join(bringup_dir, 'urdf', 'Ensamblajev12_6.urdf')
    with open(urdf_file_path, 'r') as infp:
        robot_desc = infp.read()

    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc}
        ])

    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join([
            os.path.join(bringup_dir, 'world'),
            str(Path(bringup_dir).parent.resolve())
        ])
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(bringup_dir, 'config', 'gz_bridge.yaml'),
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                )
            ]
        ),
        launch_arguments={"gz_args": ["-r -v 4 ", world]}.items(),
    )

    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "robot",
            "-topic",
            "/robot_description",
            "-x",
            "0.45409",
            "-y",
            "0.11137",
            "-z",
            "0",
            "-Y",
            "1.5708",
        ],
        output="screen",
    )


    tf_map= Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments= ["0", "0", "0", "0", "0", "0", "map", "odom"])

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    joint_group_position_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_group_position_controller', '--controller-manager', '/controller_manager'],
                output='screen',
            )],
        )
    )

    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',default_value='True',description='Use sim time if true'),
        DeclareLaunchArgument('urdf_file',default_value=os.path.join(bringup_dir, 'urdf', 'Ensamblajev12_6.urdf'),description='Whether to start RVIZ'),
        DeclareLaunchArgument('use_robot_state_pub',default_value='True',description='Whether to start the robot state publisher'),
        gz_resource_path,
        gz_sim, bridge, spawn_entity, start_robot_state_publisher_cmd, tf_map,
        joint_state_broadcaster_spawner,
        joint_group_position_controller_spawner,
    ])