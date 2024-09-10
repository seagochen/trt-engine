#!/usr/bin/env python
# coding: utf-8

import yaml


class YamlConfig:
    def __init__(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def get_mqtt_config(self):
        return self.config['mqtt']['broker']

    def get_visualization_config(self):
        return self.config['visualization']

    def get_inference_config(self):
        return self.config['inference']

    def get_record_config(self):
        return self.config['record']