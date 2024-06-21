from setuptools import setup, find_packages

setup(
    name='data_quality_checker',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for automating data quality and integrity checks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/data_quality_checker',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
