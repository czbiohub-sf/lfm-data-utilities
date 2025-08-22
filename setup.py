from setuptools import setup, find_packages

setup(
    name="lfm_data_utilities",
    description="Tools for processing data related to the lfm-scope (ulc-malaria-scope)",
    author="CZ Biohub | Bioengineering",
    author_email="paul.lebel@czbiohub.org",
    packages=find_packages("lfm-data-utilities"),
    install_requires=[
        "AllanTools==2019.9",
        "bridson==0.1.0",
        "GitPython==3.1.37",
        "matplotlib==3.5.3",
        "nbdev",
        "numpy==1.24.2",
        "opencv-python>=4.7.0.72",
        "pandas>=1.5.3",
        "tqdm>=4.64.1",
        "zarr==2.10.1",
    ],
    extras_require={
        "torch": "torch>=1.13.1",
    },
)
