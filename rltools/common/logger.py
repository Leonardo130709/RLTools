import pathlib
import typing
import tensorboard.backend.event_processing.event_accumulator as tbea
from tensorboard.backend.event_processing import tag_types


class ScalarsParser:
    def __init__(self, scalars_size: int = 0) -> None:
        self.size_guidance = {tag_types.SCALARS: scalars_size}

    def __call__(self, events_path: typing.Union[str, pathlib.Path]) -> dict[str, list[tbea.ScalarEvent]]:
        acc = tbea.EventAccumulator(
            events_path,
            size_guidance=self.size_guidance,
        )
        acc.Reload()
        return {k: acc.Scalars(k) for k in acc.scalars.Keys()}
