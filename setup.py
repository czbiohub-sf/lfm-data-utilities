from setuptools import setup, find_packages

setup(
    name="lfm-data-utilities",
    description="Tools for processing data related to the lfm-scope (ulc-malaria-scope)",
    author="CZ Biohub | Bioengineering",
    author_email="paul.lebel@czbiohub.org",
    install_requires=[
        "AllanTools==2019.9",
        "bridson==0.1.0",
        "GitPython==3.1.37",
        "matplotlib==3.5.3",
        "nbdev",
        "numcodecs==0.15.1",
        "numpy<2.0",
        "opencv-python>=4.7.0.72",
        "pandas>=1.5.3",
        "tqdm>=4.64.1",
        "zarr==2.10.1",
    ],
    extras_require={
        "torch": "torch>=1.13.1",
    },
)
