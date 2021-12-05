from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['fire']

setup(
    name='sr3',
    author="Guillaume Boniface-Chang",
    author_email="guillaume.boniface@gmail.com",
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Reimplementing the SR3 paper (partially).'
)