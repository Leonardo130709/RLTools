from abc import ABC
import dataclasses
from ruamel.yaml import YAML


@dataclasses.dataclass
class BaseConfig(ABC):
    def save(self, file_path):
        yaml = YAML()
        with open(file_path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)

    def load(self, file_path, **kwargs):
        yaml = YAML()
        with open(file_path) as f:
            config_dict = yaml.load(f)
        config_dict.update(kwargs)
        return dataclasses.replace(self, **config_dict)

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            value = field.type(value)
            setattr(self, field.name, value)