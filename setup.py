from setuptools import setup, find_packages

setup(
    name="scdc",
    version="0.0",
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'tensorflow', 'h5py', 'matplotlib', 'seaborn'
    ]
)
