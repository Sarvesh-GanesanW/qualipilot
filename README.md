# Data Quality Checker

Data Quality Checker is a Python package that automates data quality and integrity checks for your dataset. It performs several checks including missing values, duplicate rows, outliers, data type validation, and range validation. The package uses cuDF for GPU acceleration if a compatible GPU is available, and falls back to Dask for parallel processing otherwise.

## Installation

You can install the package via pip:

```bash
pip install data_quality_checker
```
## GPU Acceleration

For GPU acceleration with cuDF, please ensure your system meets the requirements listed in the cuDF installation guide (https://docs.rapids.ai/install).