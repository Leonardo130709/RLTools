import csv
import pathlib
from typing import Union, List, Dict
from tensorboard.backend.event_processing import tag_types
import tensorboard.backend.event_processing.event_accumulator as tbea
Path = Union[str, pathlib.Path]


class ScalarsParser:
    """Handles tensorboard's ScalarEvents."""
    def __init__(self, scalars_size: int = 0) -> None:
        self.size_guidance = {tag_types.SCALARS: scalars_size}

    def to_dict(self, events_path: Path) -> Dict[str, List[tbea.ScalarEvent]]:
        """Extracts scalars from tf-events file to python dictionary."""
        acc = tbea.EventAccumulator(
            events_path,
            size_guidance=self.size_guidance,
        )
        acc.Reload()
        return {k: acc.Scalars(k) for k in acc.scalars.Keys()}

    def to_csv(self, events_path: Path, output_dir: Path) -> None:
        """Write tensorboard logs to human-readable csv file.
        Output files inherit tf-events structure."""
        output_dir = pathlib.Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)

        results = self.to_dict(events_path)
        keys = set(results.keys())
        prefixes = set((k.rsplit('/', 1)[0] for k in keys))

        for prefix in prefixes:
            selected_keys = list(filter(lambda k: k.startswith(prefix), keys))
            with (output_dir / prefix).open('w') as logfile:
                writer = csv.DictWriter(logfile, fieldnames=selected_keys)
                writer.writeheader()

                for scalars in zip(*(results[k] for k in selected_keys)):
                    writer.writerow({k: sc.value for k, sc in zip(selected_keys, scalars)})
