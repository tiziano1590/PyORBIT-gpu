from setuptools import setup

# Inspired from here:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
# https://realpython.com/pypi-publish-python-package/#prepare-your-package-for-publication

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pyorbit-package",
    version='11.2.8',
    author="Luca Malavolta",
	author_email = 'luca.malavolta@unipd.it',
	url = 'https://github.com/LucaMalavolta/PyORBIT',
	packages =['pyorbit', 'pyorbit.common', 'pyorbit.classes', 'pyorbit.models', 'pyorbit.samplers', 'pyorbit.subroutines'],
	license = 'MIT',
	description ='PyORBIT: a code for exoplanet orbital parameters and stellar activity',
    long_description=long_description,
    long_description_content_type='text/markdown',
	classifiers = [
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
		'Programming Language :: Python :: 3'
		],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'pyorbit_run=pyorbit.pyorbit_run:pyorbit_run',
            'pyorbit_results=pyorbit.pyorbit_results:pyorbit_results',
        ]
    },
    zip_safe=False,
    install_requires=[
        'numpy>=1.24.3,<2',
        'numba>=0.55.2',
        'scipy>=1.8.1,<1.16',
        'matplotlib>=3.5.2,<3.11',
        'astropy>=5.1',
        'astroquery>=0.4',
        'pyerfa>=2.0',
        'emcee>=3.1.2',
        'pyyaml',
        'h5py',
        'tqdm',
        'pygtc',
        'jaxlib>=0.4.31,<=0.6.2',
        'jax>=0.4.31,<=0.6.2',
        'tinygp==0.3.0',
        'ordered-set',
        'packaging'
        #'nautilus-sampler'
        #'schwimmbad'
    ],
    extras_require={'gpu': ['jax[cuda12]==0.6.2', 'ml-dtypes<0.6', 'cupy-cuda12x>=13,<14', 'jaxoplanet==0.1.0', 'spleaf==2.1.20', 'batman-package==2.5.3']},
    setup_requires=['setuptools']
)
