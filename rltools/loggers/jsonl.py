import os
import json

from rltools.loggers import base


class JSONLogger(base.Logger):
    def __init__(self, path):
        self._path = os.path.expanduser(path)

    def write(self, metrics: base.Metrics):
        with open(self._path, "a") as logs:
            logs.write(json.dumps(metrics)+"\n")

    def close(self):
        pass
