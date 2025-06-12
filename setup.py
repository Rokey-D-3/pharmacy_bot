from setuptools import find_packages, setup
import glob
import os

package_name = 'pharmacy_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        #('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob.glob('resource/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='choin',
    maintainer_email='choin22222@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pharmacy_manager = pharmacy_bot.pharmacy_manager:main',
            'voice_input = pharmacy_bot.voice_input:main',
            'symptom_matcher = pharmacy_bot.symptom_matcher:main',
            'detector = pharmacy_bot.detector:main',
            'robot_arm = pharmacy_bot.robot_arm:main',
        ],
    },
)
