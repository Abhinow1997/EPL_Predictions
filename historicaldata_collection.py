from eplprediction.data.understat_parser.understat_api_datacollector import Understat_Parser
import asyncio 
import os
#from eplprediction.utils.path_handler import PathHandler
from loguru import logger
from eplprediction.common.configparser import ConfigParser
import argparse
from eplprediction.data.preprocessor.preprocessing import Preprocessor
import pandas as pd
from eplprediction.db_handler.sqllite_db_handler import SQLliteHandler
import aiohttp 
from aiohttp.client_exceptions import ServerDisconnectedError
import requests 
import sys

async def fetch_data_with_retry(understat_parser, season, months_of_form, output_table_name, max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        try:
            await understat_parser.get_understat_season(season=season, months_of_form=months_of_form, output_table_name=output_table_name)
            logger.success(f'Succesfully gathered and saved season {season}.')
            break  # Break out of the loop if successful
        except (aiohttp.ClientError, ServerDisconnectedError) as e:
            if attempt < max_retries - 1:
                print(f"Retry attempt {attempt + 1}/{max_retries}")
                await asyncio.sleep(retry_delay)  # Add a short delay before retrying
            else:
                logger.error(f'Max retries reached, for error {e}.')
                sys.exit(1)

def main():
    '''Parsing the configuration file'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file (e.g., config.yaml)", default='eplprediction/config/config.yaml')
    parser.add_argument("--download", action='store_true', help="Download the data from the data_co_uk website. False uses the already downloaded files.", default=False)
    config_file_path = parser.parse_args().config
    
    config_data_parser = ConfigParser(config_file_path, None)
    config_data = config_data_parser.load_and_extract_yaml()
    config = config_data_parser.load_configuration_class(config_data)
    
    logger.info(config)
    '''End of the configuration file parsing'''
    
    database_handler = SQLliteHandler(league= config.league, database= config.database)

    dataframe_list = []
    table_name_list = []
    
    # Download the data from the data_co_uk website if download is set to True
    if parser.parse_args().download:
        for season in config.seasons_to_gather:
            # Compute the two-year seas code, e.g., '1718' for '2017', '1819' for '2018'
            two_year_code = season[-2:] + str(int(season[-2:]) + 1).zfill(2)
            url_base = '/'.join(config.data_co_uk_url.split('/')[:-2])
                                                        
            # Format the URL with the t-year season code
            #url = os.path.join(url_base, two_year_code, config.data_co_uk_url.split('/')[-1])
            url = f"{url_base}/{two_year_code}/{config.data_co_uk_url.split('/')[-1]}"
            logger.debug(f"Constructed URL: {url}")

            response = requests.get(url)
            if response.status_code == 200:
                file_path = os.path.join(config.data_co_uk_path, f'E0-{season}.csv')
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w',encoding='utf-8') as file:
                    #logger.debug(response.content)
                    file.write(response.content.decode('utf-8'))
                logger.success(f'Data for season {season} saved at {file_path}.')
            else:
                logger.error(f'Failed to download data for season {season}. Response code: {response.status_code}.')                
                exit(1)

    # Iterate through each file in the directory
    for file in os.listdir(config.data_co_uk_path):
        file_path = os.path.join(config.data_co_uk_path, file)

        # Extracting table name from the file name
        season_year = int(file.split('-')[1].split('.')[0])
        table_name = f"DataCoUk_Season{season_year}_{season_year+1}"
            
        # Read data from the CSV file using pandas
        season_dataframe = pd.read_csv(file_path)
        dataframe_list.append(season_dataframe)
        table_name_list.append(table_name)

    database_handler.save_dataframes(dataframes=dataframe_list, table_names=table_name_list)    
    
    understat_parser = Understat_Parser(league = config.league, dictionary = config.data_co_uk_dictionary, database = config.database)
    
    for table_name, months_of_form in zip(['Raw_LongTermForm', 'Raw_ShortTermForm'], config.months_of_form_list):
        logger.info(f'Gathering {months_of_form} month form data for seasons in {config.seasons_to_gather}')
        for season in config.seasons_to_gather:
            asyncio.run(fetch_data_with_retry(understat_parser, season, months_of_form, table_name))
    
    
    preprocessor = Preprocessor(league=config.league, database=config.database)

    #Gathering all the seasons into two concatenated dataframes one for long term and one for short term form
    long_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_LongTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    short_term_form_season_list = preprocessor.database_handler.get_data([f'Raw_ShortTermForm_Season{season}_{str(int(season)+1)}' for season in config.seasons_to_gather])
    long_term_form_dataframe = pd.concat([dataframe for dataframe in long_term_form_season_list])
    short_term_form_dataframe = pd.concat([dataframe for dataframe in short_term_form_season_list])
    
    preprocessed_dataframes = preprocessor.preprocessing_pipeline(data=[long_term_form_dataframe, short_term_form_dataframe])
    preprocessor.database_handler.save_dataframes(dataframes=preprocessed_dataframes, table_names=['Preprocessed_LongTermForm', 'Preprocessed_ShortTermForm'])
    
if __name__ == "__main__":
    main()