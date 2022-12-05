import os
import re
from setuptools import setup, find_packages


def get_version():
	path = os.path.join(os.path.dirname(__file__), "rltools/__init__.py")
	with open(path) as f:
		version = re.search(r"__version__.*(\d+.\d+.\d+)", f.read())[1]
	return version


requirements = open("requirements.txt").readlines()
dev_requirements = open("requirements_dev.txt").readlines()

setup(
	name="rltools",
	version=get_version(),
	python_requires=">=3.9",
	install_requires=requirements,
	packages=find_packages(),
	extras_require={
		"gym": ["gym"],
		"dev": dev_requirements
	}
)
