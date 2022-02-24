from setuptools import setup, find_packages
import re

setup(
    name="isml",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    description="In-situ machine learning code.",
    install_requires=[
        "arrow",
        "imagecat>=0.4.0",
        "netcdf4",
        "numpy",
        "scipy",
        "scikit-learn",
        "sphinx",
        "sphinx_rtd_theme",
        "toyplot",
    ],
    maintainer="Warren L. Davis IV",
    maintainer_email="wldavis@sandia.gov",
    packages=find_packages(),
    scripts = [
        "bin/isml-plot-anomaly-recall",
        "bin/isml-plot-decision-summary",
        "bin/isml-generate-images",
        "bin/isml-generate-movies",
        ],
    version=re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        open(
            "isml/__init__.py",
            "r").read(),
        re.M).group(1),
)
