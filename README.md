# rltools

This package provides instruments for running Reinforcement Learning experiments.
It consists of the following components:
- [dmc_wrappers](rltools/dmc_wrappers): Wrappers for dm_env.Environment.
Thus, also applicable for dm_control suite. 
- [loggers](rltools/loggers): A module containing various loggers.
- [config.py](rltools/config.py): A class that is intended to store
and process hyperparameters.

# Installation
`rltools` can be installed by components: `dmc_wrappers`, `loggers`, `config` and
`all` to install every requirement.
For local installation use `git clone`, then:
```bash
pip install '.[dmc_wrappers, config]' 
```
or any other combination of extras.

To install directly from GitHub:
```
pip install 'rltools[all] @ git+https://github.com/lkhromykh/rltools.git'
```