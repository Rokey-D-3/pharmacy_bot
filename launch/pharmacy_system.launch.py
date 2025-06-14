#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pharmacy_bot',
            executable='pharmacy_manager',
            name='pharmacy_manager',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='voice_input',
            name='voice_input',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='symptom_matcher',
            name='symptom_matcher',
            output='screen'
        ),
        Node(
            package='pharmacy_bot',
            executable='detector',
            name='detector',
            output='screen'
        ),
        
    ])
