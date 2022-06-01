from abc import ABC
import dataclasses
from typing import TypeVar
from ruamel.yaml import YAML

Config = TypeVar('Config', bound='BaseConfig')


@dataclasses.dataclass
class BaseConfig(ABC):
    """Base class for config objects with implemented save/load functions."""
    def save(self, file_path: str) -> None:
        """Save as YAML in the specified path."""
        yaml = YAML()
        with open(file_path, 'w', encoding='utf-8') as config_file:
            yaml.dump(dataclasses.asdict(self), config_file)

    @classmethod
    def load(cls, file_path: str, **kwargs) -> Config:
        """Load config from the path."""
        yaml = YAML()
        with open(file_path, 'r', encoding='utf-8') as config_file:
            config_dict = yaml.load(config_file)
        config_dict.update(kwargs)

        fields = map(lambda field: field.name, dataclasses.fields(cls))
        config_dict = {k: v for k, v in config_dict.items() if k in fields}

        return cls(**config_dict)

    def __post_init__(self):
        """Convert fields to declared types."""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            value = field.type(value)
            setattr(self, field.name, value)
