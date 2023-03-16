import os
import re
from setuptools import setup, find_packages


def get_version():
	path = os.path.join(os.path.dirname(__file__), "rltools/__init__.py")
	with open(path) as f:
		version = re.search(r"__version__.*(\d+.\d+.\d+)", f.read())[1]
	return version


_dmc_wrappers_req = ["dm_env"]
_loggers_req = ["tensorflow"]
_config_req = ["ruamel.YAML"]
_all_req = _dmc_wrappers_req + _loggers_req + _config_req

setup(
	name="rltools",
	version=get_version(),
	python_requires=">=3.9",
	install_requires=["numpy"],
	packages=find_packages(),
	extras_require={
		"dmc_wrappers": _dmc_wrappers_req,
		"loggers": _loggers_req,
		"config": _config_req,
		"all": _all_req
	}
)
