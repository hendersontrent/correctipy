from setuptools import setup, find_packages

with open('README.md') as file:
    long_description = file.read()

install_requires = [
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'scipy==1.5.0',
]

testing_extras = [
    'pytest==5.4.2',
]

setup(
    name='correctipy',
    packages=find_packages(),
    version='0.1.0',
    description='Network analysis for time series',
    author='Trent Henderson',
    author_email='then6675@uni.sydney.edu.au',
    url='https://github.com/hendersontrent/correctipy',
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    install_requires=install_requires,
    extras_require={'testing': testing_extras}
)