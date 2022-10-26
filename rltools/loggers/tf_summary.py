from typing import Optional
try:
    import tensorflow as tf
except ImportError:
    pass

from rltools.loggers import base


class TFSummaryLogger(base.Logger):
    """Tensorboard logger"""
    def __init__(
            self,
            logdir: str,
            label: str,
            step_key: Optional[str] = None
    ):
        self._iter = 0
        self._label = label
        self._summary = tf.summary.create_file_writer(logdir)
        self._step_key = step_key

    def write(self, metrics: base.Metrics):
        step = metrics[self._step_key] if self._step_key else self._iter

        with self._summary.as_default():
            for key in metrics.keys() - [self._step_key]:
                tf.summary.scalar(
                    f'{self._label}/{key}',
                    data=metrics[key],
                    step=step
                )
        self._iter += 1

    def close(self):
        self._summary.close()
