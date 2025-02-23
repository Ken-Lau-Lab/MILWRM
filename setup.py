# -*- coding: utf-8 -*-
"""
package setup
"""
import os
import io
import setuptools
from setuptools import setup


f_compile_args = ["-ffixed-form", "-fdefault-real-8"]


def read(fname):
    with io.open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as _in:
        return _in.read()


if __name__ == "__main__":
    import versioneer

    with open("README.rst", "r") as fh:
        long_description = fh.read()

    setup(
        name="MILWRM",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description="Multiplex Image Labeling With Regional Morphology",
        long_description=long_description,
        author="Cody Heiser",
        author_email="cody.n.heiser@vanderbilt.edu",
        url="https://github.com/codyheiser/MILWRM",
        install_requires=read("requirements.txt").splitlines(),
        packages=setuptools.find_packages(exclude=["tutorials"]),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
        ],
        python_requires=">=3.6",
    )
