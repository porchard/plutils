import glob
import re
from setuptools import setup, find_packages

with open("plutils/__init__.py") as f:
    __version__ = re.search(r'__version__ ?= ?[\'\"]([\w.]+)[\'\"]', f.read()).group(1)

# Setup information
setup(
    name = 'plutils',
    version = __version__,
    packages = find_packages(),
    description = 'Utility functions I commonly use for my work in the Parker lab.',
    author = 'Peter Orchard',
    author_email = 'porchard@umich.edu',
    scripts = glob.glob('bin/*'),
    install_requires = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'pyarrow',
        'pysam',
        'pybedtools',
        'sklearn']
)
