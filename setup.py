from glob import glob
from os.path import basename
from os.path import splitext

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def _requires_from_file(filename):
    return [string.replace("\n", "") for string in open(filename).readlines()]


setuptools.setup(
    name="reservoir-network",
    version="0.0.1",
    license="MIT License",
    description="リザバーコンピューティング用のユーティリティ関数",
    author="Soh Ohara",
    url="https://github.com/wildgeece96/reservoir_demo",
    packages=setuptools.find_packages("reservoir_network"),
    package_dir={"": "reservoir_network"},
    py_modules=[
        splitext(basename(path))[0]
        for path in glob("reservoir_network/**/*.py", recursive=True)
    ],
    zip_safe=False,
    install_requires=_requires_from_file("requirements.txt"),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"])
