import abc
from typing import Mapping, Any

Metrics = Mapping[str, Any]


class Logger(abc.ABC):
    """Handles metrics stream."""

    @abc.abstractmethod
    def write(self, metrics: Metrics) -> None:
        """Write metrics to destination."""

    @abc.abstractmethod
    def close(self) -> None:
        """Flush data and close a logger."""
