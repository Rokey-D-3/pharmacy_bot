#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pharmacy_bot',
            executable='pharmacy_manager.py',
            name='pharmacy_manager',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='voice_input.py',
            name='voice_input',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='symptom_matcher.py',
            name='symptom_matcher',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='detector.py',
            name='detector',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='robot_arm.py',
            name='robot_arm',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='pharmacy_gui.py',
            name='pharmacy_manager',
            output='screen'
        ),
        
    ])
