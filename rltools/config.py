from typing import Optional, get_origin

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
        yaml = YAML(typ="safe", pure=True)
        with open(file_path, "w", encoding="utf-8") as config_file:
            yaml.dump(dataclasses.asdict(self), config_file)

    @classmethod
    def load(cls, file_path: str, **kwargs) -> "Config":
        """Load config from a YAML. Values can by updated by kwargs."""
        yaml = YAML(typ="safe", pure=True)
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
            if hasattr(field.type, "__args__"):
                value = _topy(map(_topy(field.type.__args__[0]), value))
            else:
                value = _topy(field.type)(value)
            setattr(self, field.name, _topy(field.type)(value))

    @classmethod
    def from_entrypoint(cls,
                        parser: Optional[argparse.ArgumentParser] = None
                        ) -> "Config":
        """Populate commandline kwargs with a config fields."""
        # TODO: proper container handler.
        def arg_help(field):
            match = re.search(fr"\s{field.name}: (.*)\n", cls.__doc__)
            if match:
                return match[1]
            return None

        def get_type(field):
            # TODO: proper bool handler.
            is_tuple = isinstance(field.default, tuple)
            if is_tuple:
                t = _topy(field.type.__args__[0])
            else:
                t = _topy(field.type)
            if t is bool:
                t = int
            return t, is_tuple


        if parser is None:
            parser = argparse.ArgumentParser()
        known_args = set()

        for field in dataclasses.fields(cls):
            known_args.add(field.name)
            dtype, is_tuple = get_type(field)
            parser.add_argument(
                f"--{field.name}",
                type=dtype,
                default=field.default,
                help=arg_help(field),
                nargs="+" if is_tuple else "?",
            )
        args, _ = parser.parse_known_args()
        kwargs = {k: v for k, v in vars(args).items() if k in known_args}
        return cls(**kwargs)


def _topy(dtype):
    """To support both typing hints and GenericAliases."""
    return get_origin(dtype) or dtype
