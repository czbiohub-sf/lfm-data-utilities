from setuptools import setup, find_packages

setup(
    name="lfm_data_utilities",
    version="1.0",
    description="Tools for processing data related to the lfm-scope (ulc-malaria-scope)",
    author="CZ Biohub | Bioengineering",
    author_email="paul.lebel@czbiohub.org",
    packages=find_packages(),
    install_requires=[
        "AllanTools==2019.9",
        "matplotlib>=3.5.3",
        "numpy>=1.24.2",
        "opencv-python>=4.7.0.72",
        "tqdm>=4.64.1",
        "opencv-python>=4.7.0.72",
        "zarr==2.13.3"
    ],
)
