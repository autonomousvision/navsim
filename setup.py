import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Installs
setuptools.setup(
    name="navsim",
    version="1.1.0",
    author="University of Tuebingen",
    author_email="kashyap.chitta@uni-tuebingen.de",
    description="NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking",
    url="https://github.com/autonomousvision/navsim",
    python_requires=">=3.9",
    packages=setuptools.find_packages(script_folder),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
