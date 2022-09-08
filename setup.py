from setuptools import setup, find_packages
import platform


REQUIREMENTS = [
    'numpy', 'scipy', 'h5py', 'matplotlib', 'seaborn'
]

if platform.system() == 'Darwin':
    REQUIREMENTS.append('tensorflow-macos')
else:
    REQUIREMENTS.append('tensorflow')


setup(
    name="scdc",
    version="0.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS
)
