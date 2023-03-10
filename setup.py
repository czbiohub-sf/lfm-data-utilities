from setuptools import setup

setup(
    name="LFM Data Utilities",
    version="1.0",
    description="Tools for processing data related to the lfm-scope (ulc-malaria-scope)",
    author="CZ Biohub | Bioengineering",
    author_email="paul.lebel@czbiohub.org",
    packages=[],
    install_requires=[
        "matplotlib>=3.5.3",
        "tqdm>=4.64.1",
        "AllanTools==2019.9",
        "numpy==1.24.2",
        "matplotlib==3.7.0",
        "opencv-python>=4.7.0.72",
    ],
)
