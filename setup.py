from setuptools import setup, find_packages

setup(
    name="rsgn",
    version="0.1.0",
    description="Resonant Sparse Geometry Networks",
    author="Hasi Hays",
    author_email="hasi.hays@research.org",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
