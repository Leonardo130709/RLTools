from abc import ABC
import dataclasses
from ruamel.yaml import YAML


@dataclasses.dataclass
class BaseConfig(ABC):
    """Base config class with implemented save/load functions."""
    def save(self, file_path: str) -> None:
        yaml = YAML()
        with open(file_path, 'w', encoding='utf-8') as config_file:
            yaml.dump(dataclasses.asdict(self), config_file)

    @classmethod
    def load(cls, file_path: str, **kwargs) -> cls:
        """Load config from the path."""
        yaml = YAML()
        with open(file_path, 'r', encoding='utf-8') as config_file:
            config_dict = yaml.load(config_file)
        config_dict.update(kwargs)

        fields = map(lambda field: field.name, dataclasses.fields(cls))
        config_dict = {k: v for k, v in config_dict.items() if k in fields}

        return cls(**config_dict)

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            value = field.type(value)
            setattr(self, field.name, value)
