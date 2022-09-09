import abc
import dataclasses

from ruamel.yaml import YAML


@dataclasses.dataclass
class Config(abc.ABC):
    """Config object with all the hyperparameters."""

    def save(self, file_path: str):
        """Save as YAML in a specified path."""
        yaml = YAML()
        with open(file_path, "w", encoding="utf-8") as config_file:
            yaml.dump(dataclasses.asdict(self), config_file)

    @classmethod
    def load(cls, file_path: str, **kwargs) -> "Config":
        """Load config from a YAML."""
        yaml = YAML()
        with open(file_path, "r", encoding="utf-8") as config_file:
            config_dict = yaml.load(config_file)
        config_dict.update(kwargs)

        fields = map(lambda field: field.name, dataclasses.fields(cls))
        config_dict = {k: v for k, v in config_dict.items() if k in fields}

        return cls(**config_dict)

    def __post_init__(self):
        """Casts fields to declared types."""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            setattr(self, field.name, field.type(value))
