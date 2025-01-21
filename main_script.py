from eplprediction.data.understat_parser.understat_api_datacollector import Understat_Parser
from eplprediction.common.configparser import ConfigParser
from eplprediction.data.nextmatches import NextMatchScheduler
from eplprediction.data.preprocessor import Preprocessor
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

    understat_parser = Understat_Parser(league = config.league, dictionary = config.data_co_uk_dictionary, database=config.database)
    upcoming_match_scheduler = NextMatchScheduler(
        league = config.league,
        current_season = config.current_season,
        months_of_form_list= config.months_of_form_list,
        data_co_uk_ulr= config.data_co_uk_url, 
        data_co_uk_dict= config.data_co_uk_dictionary, 
        fixtures_url = config.fixture_download_url,
        fixtures_dict = config.fixture_download_dictionary,
        database = config.database
        )
    #upcoming_match_scheduler.update_dataset('odds')
    upcoming_match_scheduler.update_dataset('fixtures')
    upcoming_match_scheduler.setup_upcoming_fixtures()

    #Updating the UpcomingShortTerm and UpcomingLongTerm tables
    for name, months_of_form in zip(['Raw_LongTermForm', 'Raw_ShortTermForm'], config.months_of_form_list):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(understat_parser.get_understat_season(season = config.current_season, months_of_form = months_of_form, output_table_name=name))
    
    #Preprocessing season raw statistics   
    preprocessor = Preprocessor(league=config.league, database=config.database)
    long_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_LongTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    short_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_ShortTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    long_term_form_dataframe = pd.concat([dataframe for dataframe in long_term_form_season_list])
    short_term_form_dataframe = pd.concat([dataframe for dataframe in short_term_form_season_list])
    
    preprocessed_dataframes = preprocessor.preprocessing_pipeline(data=[long_term_form_dataframe, short_term_form_dataframe])
    preprocessor.database_handler.save_dataframes(dataframes=preprocessed_dataframes, table_names=['Preprocessed_LongTermForm', 'Preprocessed_ShortTermForm'])
    
if __name__ == "__main__":
    main()