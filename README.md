# DataWhiz

DataWhiz is a Python package that automates data quality and integrity checks for your dataset. It performs several checks including missing values, duplicate rows, outliers, data type validation, and range validation. The package uses cuDF for GPU acceleration if a compatible GPU is available, and falls back to Dask for parallel processing otherwise.

## Installation

### Basic Installation

You can install the package via pip:

```bash
pip install datawhiz
```

## Installation with GPU Support
To use GPU acceleration with cuDF, you need to set up a compatible environment. Follow these steps:

### Create a conda environment with RAPIDS:

```bash
conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia \
    rapids=24.06 python=3.11 cuda-version=12.2
```
### Activate the conda environment:

```bash
conda activate rapids-24.06
```
### Install DataWhiz in the conda environment:

```bash
pip install datawhiz
```

Check the rapids website for cuDF installation. (https://docs.rapids.ai/install)