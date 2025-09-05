#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script para instalar el paquete data_science_project."""

from setuptools import setup, find_packages

setup(
    name="data_science_project",
    version="1.0.0",
    author="Universidad Autónoma de Occidente",
    description="Proyecto de detección de neumonía usando Deep Learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "tensorflow>=2.8.0",
        "pillow>=8.0.0",
        "pydicom>=2.3.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pytest>=7.0.0"
    ]
)
