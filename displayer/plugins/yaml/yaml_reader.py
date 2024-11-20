import yaml
from dataclasses import dataclass


@dataclass
class MQTTConfig:
    host: str
    port: int


@dataclass
class ClientConfig:
    id: str


@dataclass
class TopicsConfig:
    inference: str
    stream: str
    publish_by: str


@dataclass
class InferenceConfig:
    label_path: str
    width: int
    height: int

@dataclass
class DisplayConfig:
    width: int
    height: int
    auto_resize: bool
    show_fps: bool
    show_inference_info: bool
    show_local_time: bool


@dataclass
class RecordConfig:
    enable: bool
    filename: str


@dataclass
class Config:
    mqtt: MQTTConfig
    client: ClientConfig
    topics: TopicsConfig
    inference: InferenceConfig
    display: DisplayConfig
    record: RecordConfig


def load_config(file_path: str) -> Config:
    """Load YAML configuration into Python objects."""
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    return Config(
        mqtt=MQTTConfig(**config_dict['mqtt']),
        client=ClientConfig(**config_dict['client']),
        topics=TopicsConfig(**config_dict['topics']),
        inference=InferenceConfig(**config_dict['inference']),
        display=DisplayConfig(**config_dict['display']),
        record=RecordConfig(**config_dict['record'])
    )


def save_config(config: Config, file_path: str):
    """Save Python configuration object back to a YAML file."""
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
