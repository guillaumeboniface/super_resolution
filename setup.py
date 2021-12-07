from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["fire"]
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sr3",
    author="Guillaume Boniface-Chang",
    author_email="guillaume.boniface@gmail.com",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Reimplementing the super resolution paper (partially).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.7"
)