import pandas as pd
import numpy as np
import pickle
import feather
import gzip
import yaml
import re
import os
import sys

from pydantic import BaseModel
from jinja2 import Template
import concurrent.futures
import time
import json
from datetime import datetime, timedelta
from loguru import logger


def format_dates(date_input):
    """
    This function takes a date input which could be 'latest', a single date,
    or a date range, and returns a list of dates in the datetime format.
    """
    today_date = datetime.today()

    if date_input.lower() == "latest":
        return [today_date]
    
    date_parts = date_input.split("|")
    
    if len(date_parts) == 1:
        return [datetime.strptime(date_parts[0], '%d-%m-%Y')]
    
    elif len(date_parts) == 2:
        return [
            datetime.strptime(date_parts[0], '%d-%m-%Y'),
            datetime.strptime(date_parts[1], '%d-%m-%Y')
        ]
    
    else:
        raise ValueError("Invalid date input format")
    

def save(data, filename):
    folders = os.path.dirname(filename)
    if folders:
        os.makedirs(folders, exist_ok=True)

    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            # Since feather doesn't support writing to the file handle, we
            # can't easily point it to gzip.
            raise NotImplementedError(
                "Saving to compressed .feather not currently supported."
            )
        else:
            fp = gzip.open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            if str(type(data)) != "<class 'pandas.core.frame.DataFrame'>":
                raise TypeError(
                    ".feather format can only be used to save pandas "
                    "DataFrames"
                )
            feather.write_dataframe(data, filename)
        else:
            fp = open(filename, "wb")
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    """
    Loads data saved with save() (or just normally saved with pickle).
    Autodetects gzip if filename ends in '.gz'
    Also reads feather files denoted .feather or .fthr.

    Parameters
    ----------
    filename -- String with the relative filename of the pickle/feather
    to load.
    """
    fl = filename.lower()
    if fl.endswith(".gz"):
        if fl.endswith(".feather.gz") or fl.endswith(".fthr.gz"):
            raise NotImplementedError("Compressed feather is not supported.")
        else:
            fp = gzip.open(filename, "rb")
            return pickle.load(fp)
    else:
        if fl.endswith(".feather") or fl.endswith(".fthr"):
            import feather

            return feather.read_dataframe(filename)
        else:
            fp = open(filename, "rb")
            return pickle.load(fp)


def read_yaml(filename, render=False, **kwargs):
    """
    Read yaml configuation and returns a dict

    Parameters
    ----------
    filename: string
        Path including yaml file name
    render: Boolean, default = False
        Template rendering
    **kwargs:
        Template render args to be passed
    """
    if render:
        yaml_text = Template(open(filename, "r").read())
        yaml_text = yaml_text.render(**kwargs)
        config = yaml.safe_load(yaml_text)
    else:
        with open(filename) as f:
            config = yaml.safe_load(f)

    return config


def setup_logger(name, log_path):
    """
    Setup loguru logger with file and console handlers.

    Parameters
    ----------
    name: string
        Name for the log file (e.g., "linear_regression", "knn")
    log_path: Path
        Path to the logs directory
    """
    container_date = datetime.now().strftime("%Y-%m-%d")
    logger.remove()
    logger.add(
        os.path.join(log_path, f"{name}_{container_date}.log"),
        format="<green>{time}</green> | <yellow>{name}</yellow> | {level} | <cyan>{message}</cyan>",
    )
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time}</green> | <yellow>{name}</yellow> | {level} | <cyan>{message}</cyan>",
    )
