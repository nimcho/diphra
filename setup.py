
import numpy
import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages, Extension

setup(
    name='diphra',
    version='0.1.0',
    description='Extraction and Interpretation of Phrases',
    author='Lukas Banic',
    author_email='lukas.banic@protonmail.com',
    url='https://github.com/nimcho/diphra',
    license='LGPL-2.1',
    ext_modules=[
        Extension(
            'diphra.models.pocs2vec_inner',
            sources=['./diphra/models/pocs2vec_inner.c'],
            include_dirs=[numpy.get_include()]
        )
    ],
    packages=find_packages(),
    zip_safe=False,
    setup_requires=[
        'numpy >= 1.3'
    ],
    install_requires=[
        'numpy >= 1.3',
        'scipy >= 0.7.0',
        'more_itertools',
        'sortedcontainers',
    ],
    extras_require={
        'h5py': ['h5py'],
    },
    include_package_data=True,
)

