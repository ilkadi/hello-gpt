import yaml
import pprint
import os

class ConfigHelper:    
    def update_with_config(self, config, extension_config):
        print(f"Updating with extension config {extension_config} ..")
        if extension_config is not None:
            config_ext_path = os.path.join(extension_config)
            with open(config_ext_path, 'r') as file:
                config.update(yaml.safe_load(file))    
        return config
    
    def print_config(self, config):
        print("Config:")
        pprint.pprint(config)   