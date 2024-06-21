from setuptools import setup, find_packages

setup(
    name='data_pilot_checker',  
    version='1',  
    packages=find_packages(),
    install_requires=[
        'pandas',
        'dask',
        'dask[dataframe]',
    ],
    author='Sarvesh Ganesan',
    author_email='sarveshganesanwork@gmail.com',
    description='A package for automating data quality and integrity checks with optional GPU acceleration using cuDF',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Sarvesh-GanesanW/datapilot',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
