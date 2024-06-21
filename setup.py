from setuptools import setup, find_packages

setup(
    name='datawhiz',  # Updated package name
    version='0.1.1',  # Updated version number
    packages=find_packages(),
    install_requires=[
        'pandas',
        'dask[dataframe]',
    ],
    author='Sarvesh Ganesan',
    author_email='sarveshganesanwork@gmail.com',
    description='A package for automating data quality and integrity checks with optional GPU acceleration using cuDF',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Sarvesh-GanesanW/datawhiz',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
