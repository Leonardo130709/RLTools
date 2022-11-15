from typing import Optional

import abc
import re
import argparse
import dataclasses

from ruamel.yaml import YAML


# TODO: froze it but with types conversion.
@dataclasses.dataclass
class Config(abc.ABC):
    """Config object with all the hyperparameters."""

    def save(self, file_path: str):
        """Save as YAML in a specified path."""
        yaml = YAML(typ='safe', pure=True)
        with open(file_path, "w", encoding="utf-8") as config_file:
            yaml.dump(dataclasses.asdict(self), config_file)

    @classmethod
    def load(cls, file_path: str, **kwargs) -> "Config":
        """Load config from a YAML. Values can by updated by kwargs."""
        yaml = YAML(typ='safe', pure=True)
        with open(file_path, "r", encoding="utf-8") as config_file:
            config_dict = yaml.load(config_file)
        config_dict.update(kwargs)

        known_names = map(lambda field: field.name, dataclasses.fields(cls))
        config_dict = {k: v for k, v in config_dict.items() if k in known_names}

        return cls(**config_dict)

    def __post_init__(self):
        """Casts fields to declared types."""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            setattr(self, field.name, field.type(value))

    @classmethod
    def as_entrypoint(cls,
                      parser: Optional[argparse.ArgumentParser] = None
                      ) -> argparse.ArgumentParser:
        """Populate commandline kwargs with a config fields."""
        # TODO: proper container handler.
        def arg_help(name):
            match = re.search(fr"\s{name}: (.*)\n", cls.__doc__)
            if match:
                return match[1]
            return None

        if parser is None:
            parser = argparse.ArgumentParser()
        for field in dataclasses.fields(cls):
            is_container = isinstance(field.default, tuple)
            parser.add_argument(
                f"--{field.name}",
                type=field.type.__args__[0] if is_container else field.type,
                default=field.default,
                help=arg_help(field.name),
                nargs="+" if is_container else None,
            )
        return parser
