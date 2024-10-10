import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare the argument for the config file
    model_fp = LaunchConfiguration("model_fp")

    params_fp = os.path.join(
        get_package_share_directory("maxent_irl_maps"),
        "config",
        "ros",
        "default_params.yaml",
    )

    nodes = [
        # Declare the use_sim_time argument
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),
        # Declare the launch argument with a default value
        DeclareLaunchArgument(
            "model_fp",
            default_value="",
            description="Path to maxent irl model",
        ),
        # Node definition
        Node(
            package="maxent_irl_maps",
            executable="cvar_maxent_irl",
            name="cvar_maxent_irl_node",
            output="screen",
            parameters=[
                {"model_fp": model_fp},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
                {"models_dir": LaunchConfiguration("models_dir")},
                params_fp,
            ],
        ),
    ]

    return LaunchDescription(nodes)
