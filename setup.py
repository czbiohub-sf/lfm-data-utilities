from setuptools import setup, find_packages

setup(
    name="lfm_data_utilities",
    description="Tools for processing data related to the lfm-scope (ulc-malaria-scope)",
    author="CZ Biohub | Bioengineering",
    author_email="paul.lebel@czbiohub.org",
    packages=find_packages("lfm-data-utilities"),
    install_requires=[
        "matplotlib>=3.5.3",
        "tqdm>=4.64.1",
        "AllanTools==2019.9",
        "GitPython==3.1.31",
        "numpy==1.24.2",
        "pandas>=1.5.3",
        "matplotlib==3.7.0",
        "opencv-python>=4.7.0.72",
        # "autofocus @ git+https://github.com/czbiohub/ulc-malaria-autofocus@main",
        "zarr",
    ],
    extras_require={
        "torch": "torch>=1.13.1",
    },
)
