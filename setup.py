from setuptools import find_packages, setup
import os
from glob import glob

package_name = "maxent_irl_maps"
data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name), glob("launch/*.py")),
    (os.path.join("share", package_name, "config", "ros"), glob("config/ros/*.yaml")),
]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="striest",
    maintainer_email="striest@andrew.cmu.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cvar_maxent_irl = maxent_irl_maps.gridmap_to_cvar_costmap_node:main",
        ],
    },
)
