from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'rh56_controller'
data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
]

# Install MuJoCo scene assets for ros2 launch/runtime resolution.
for root, _, files in os.walk('h1_mujoco'):
    if not files:
        continue
    rel_root = os.path.relpath(root, '.')
    install_dir = os.path.join('share', package_name, rel_root)
    data_files.append((install_dir, [os.path.join(root, f) for f in files]))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['rh56_controller', 'rh56_controller.*'], exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools', 'pyserial', 'numpy'],
    zip_safe=True,
    maintainer='William Xie',
    maintainer_email='wixi6454@colorado.edu',
    description='ROS 2 driver for the Inspire RH56DFX hand.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rh56_driver = rh56_controller.rh56_driver:main',
            'hand_bridge_node = rh56_controller.hand_bridge_node:main',
            'grasp_viz = rh56_controller.grasp_viz:main',
            'motion_arbiter_node = rh56_controller.motion_arbiter_node:main',
        ],
    },
)
