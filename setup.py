from setuptools import setup, find_packages

setup(
    name='WoodsPY',
    version='1.0',
    packages=find_packages(),
    description='Woods Hole 2024 code base',
    author='Guillaume Le Treut',
    author_email='guillaume.letreut@gmail.com',
    install_requires=[
      'numpy',
      'scipy',
      'pandas',
      'tabulate',
      'matplotlib',
      'jupyterlab'
      ],
    )

