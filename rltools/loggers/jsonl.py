from typing import TextIO, Union
import json

from . import base


class JSONLogger(base.Logger):
    def __init__(self, log_file: Union[str, TextIO]):
        if isinstance(log_file, str):
            self._file = open(log_file, "a")
        else:
            self._file = log_file

    def write(self, metrics: base.Metrics):
        line = json.dumps(metrics)+"\n"
        self._file.write(line)

    def close(self):
        self._file.flush()
        self._file.close()
