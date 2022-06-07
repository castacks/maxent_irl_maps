from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['maxent_irl_costmaps', 'maxent_irl_costmaps.algos', 'maxent_irl_costmaps.costmappers', 'maxent_irl_costmaps.networks', 'maxent_irl_costmaps.dataset'],
  package_dir={'': 'src'}
)

setup(**d)
