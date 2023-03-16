from typing import Optional

import tensorflow as tf

from rltools.loggers import base


class TFSummaryLogger(base.Logger):
    """Tensorboard logger"""

    def __init__(
            self,
            logdir: str,
            label: str,
            step_key: Optional[str] = None
    ) -> None:
        self._iter = 0
        self._label = label
        self._summary = tf.summary.create_file_writer(logdir)
        self._step_key = step_key

    def write(self, metrics: base.Metrics) -> None:
        step = metrics.pop(self._step_key) if self._step_key else self._iter

        with self._summary.as_default(step=step):
            for key, value in metrics.items():
                name = "/".join([self._label, key])
                lvs = len(value.shape)
                if lvs == 0:
                    tf.summary.scalar(name, value)
                elif lvs == 1:
                    tf.summary.histogram(name, value)
                elif lvs in (2, 3):
                    tf.summary.image(name, value)
                else:
                    raise NotImplementedError(name, value)
        self._iter += 1

    def close(self) -> None:
        self._summary.close()
