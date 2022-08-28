import json
import pathlib

from . import base


class JSONLogger(base.Logger):
    def __init__(self, path):
        self._path = pathlib.Path(path).expanduser()

    def write(self, metrics: base.Metrics):
        with self._path.open("a") as logs:
            logs.write(json.dumps(metrics)+"\n")

    def close(self):
        pass
