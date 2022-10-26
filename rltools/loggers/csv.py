from typing import TextIO, Union
import csv

from rltools.loggers import base


class CSVLogger(base.Logger):

    def __init__(self,
                 log_file: Union[str, TextIO],
                 ):
        if isinstance(log_file, str):
            self._file = open(log_file, "a")
        else:
            self._file = log_file

        self._writer = None

    def write(self, metrics):
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=metrics.keys(),
                extrasaction="ignore"
            )
        if not self._file.tell():
            self._writer.writeheader()

        self._writer.writerow(metrics)

    def close(self):
        self._file.flush()
        self._file.close()
        self._writer = None
