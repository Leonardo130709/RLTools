from setuptools import setup, find_packages

setup(
	name="rltools",
	version="0.0.1",
	install_requires=open("requirements.txt").readlines(),
	packages=find_packages()
)
