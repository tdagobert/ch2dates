"""
...
"""
from os.path import join, abspath, dirname
from setuptools import setup

here = abspath(dirname(__file__))

PACKAGE = "ch2dates"

about = {}
file_about = join(here, PACKAGE, "__about__.py")
with open(file_about, mode="r", encoding="utf-8") as fic:
    exec(fic.read(), about)

def readme():
    """
    ...
    """
    file_readme = join(here, "README.md")
    with open(file_readme, mode="r", encoding="utf-8") as fichier:
        return fichier.read()

requirements = [
    'joblib>=1.4.2',
    'numpy>=1.26.4',
    'scipy>=1.12.0',
    'matplotlib>=3.8.3',
    'iio>=16',
    'numba>=0.59.0'
]

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=readme(),
    long_description_content_type='text/markdown',
    url=about["__url__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    packages=[PACKAGE],
    install_requires=requirements,
    python_requires=">=3.5",
)
