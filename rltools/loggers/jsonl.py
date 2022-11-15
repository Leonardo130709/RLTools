import os
import json

from rltools.loggers import base


class JSONLogger(base.Logger):
    """Save as .jsonl"""

    def __init__(self, path: str):
        self._path = os.path.expanduser(path)

    def write(self, metrics: base.Metrics):
        with open(self._path, "a", encoding="utf-8") as logs:
            logs.write(json.dumps(metrics)+"\n")

    def close(self):
        pass
