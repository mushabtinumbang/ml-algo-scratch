import os
from pathlib import Path
import src.utilities.utils as utils

# Paths
root_path = Path(__file__).parent.parent.parent.resolve()
config_path = root_path / "config"
img_path = root_path / "img"
raw_data_path = root_path / "data" / "raw"
predicted_data_path = root_path / "data" / "predicted"
sql_data_path = root_path / "data" / "sql"
log_path = root_path / "logs"

class ConfigManager(object):
    """
    Config Manager to manage main configurations
    and store them as attributes depending on
    the environment
    """

    def __init__(self, config_file="main_config.yaml"):
        # load main_config
        self.params = utils.read_yaml(os.path.join(config_path, "main_config.yaml"))