from eplprediction.common.configparser import ConfigParser
from loguru import logger
import asyncio
import os
import argparse
import pandas as pd

"""
English Premier League Predictor 

This script performs data scraping and processing to predict outcomes in the specified English Premier League configuration.
"""

def main():
    """
    Main entry point for this script.
    This function coordinates the entire data scraping and processing pipeline.
    """
    #Parsing the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to configuration file (e.g., config.yaml)", default='eplprediction/config/config.yaml')
    config_file_path = parser.parse_args().config

    # Loading and extracting configuration data
    config_data_parser = ConfigParser(config_file_path, None)
    config_data = config_data_parser.load_and_extract_yaml()
    config = config_data_parser.load_configuration_class(config_data)

    logger.info(config)
    logger.debug("Configuration File loaded")

if __name__ == "__main__":
    main()