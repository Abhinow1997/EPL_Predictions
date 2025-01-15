import yaml 
from loguru import logger 
from dataclasses import dataclass
import sys
import importlib
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.svm import SVR

class ConfigParser:

    def __init__(self, config_path, section_name=None) -> None:
        self.config_path = config_path
        self.section_name = section_name

    def load_and_extract_yaml(self, path=None) -> dict:
        if path is None:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            if self.section_name is not None and self.section_name in config_data:
                logger.success(f'Successfully loaded {self.section_name} section from config.yaml')
                return config_data[self.section_name]
            if self.section_name is None:
                logger.success(f'Successfully loaded the config.yaml')
                return config_data
            else:
                logger.error(f'Section {self.section_name} not available in the config.yaml file')
                return None  # Section not found in the YAML file
        else:
            with open(path, 'r') as file:
                config = yaml.safe_load(file)
            
            logger.success(f'Successfully loaded {path}')
            return config          
        
    def yamlvalidation(self, config_data) -> None:
        if config_data['model']['voting']['form_longterm_percentage'] + config_data['model']['voting']['form_shortterm_percentage'] != 1:
            logger.error('Form  weights should add upto 1! Please check configuration file!')
            sys.exit(1)
        if config_data['model']['regressor'] not in ['LinearRegression', 'PoissonRegressor', 'SVR']:
            logger.error(f"Selected Model {config_data['model']['regressor']} is incorrect!! Configuration options needs to be updated!")
            sys.exit(1)
        if config_data['league'] not in ['EPL', 'La_Liga', 'Bundesliga', 'Serie_A', 'Ligue_1']:
            logger.error(f"League {config_data['league']} is invalid! Please check configuration file!")
            sys.exit(1)


    def load_configuration_class(self, config_data) -> dataclass:
        """Calls yamlvalidation(). Imports the regressor depending on the users choice. Loads self.config dataclass with the validated configuration settings.

        Args:
            config_data (_type_: dict): The configuration dictionary in the format output from load_and_extract_yaml().

        Returns:
            self.config (_type_: Configuration_dataclass_object): The dataclass with the validated configuration settings.
        """
        self.yamlvalidation(config_data)
        if (config_data['model']['regressor'] == "LinearRegression") or (config_data['model']['regressor'] == "PoissonRegressor"):
            module_path = 'sklearn.linear_model'
            regressor_module = importlib.import_module(module_path)
            regressor = getattr(regressor_module, config_data['model']['regressor'])
        if config_data['model']['regressor'] == 'SVR':
            module_path = 'sklearn.svm'
            regressor_module = importlib.import_module(module_path)
            regressor = getattr(regressor_module, config_data['model']['regressor'])
        
        self.config = Configuration(
                league= config_data['league'],
                regressor = regressor,
                bettor_bank = config_data['bettor']['initial_bank'],
                bettor_kelly_cap = config_data['bettor']['kelly_cap'],
                evaluation_output = config_data['data_gathering']['paths'][config_data['league']]['evaluation_output'],
                months_of_form_list = [config_data['data_gathering']['long_term_form'], config_data['data_gathering']['short_term_form']],
                seasons_to_gather= config_data['data_gathering']['seasons_to_gather'],
                current_season= config_data['data_gathering']['current_season'],
                data_co_uk_path= config_data['data_gathering']['paths'][config_data['league']]['data_co_uk_path'],
                database = config_data['data_gathering']['paths'][config_data['league']]['database'],
                data_co_uk_url= config_data['data_gathering']['data_co_uk'][config_data['league']]['url'],
                data_co_uk_dictionary= self.load_and_extract_yaml(path = config_data['data_gathering']['data_co_uk'][config_data['league']]['dictionary_path']),
                fixture_download_url= config_data['data_gathering']['fixture_download'][config_data['league']]['url'],
                fixture_download_dictionary= self.load_and_extract_yaml(path = config_data['data_gathering']['fixture_download'][config_data['league']]['dictionary_path']),
                voting_dict= { 'long_term': config_data['model']['voting']['form_longterm_percentage'], 'short_term': config_data['model']['voting']['form_shortterm_percentage']},
                matchdays_to_drop = config_data['model']['matchdays_to_drop']
        )

        return self.config
    

@dataclass
class Configuration:
    """A dataclass containing the configuration settings.
    """
    league: str
    regressor: object
    bettor_bank: float
    bettor_kelly_cap: float
    evaluation_output: str
    months_of_form_list: list
    database: str
    seasons_to_gather: list
    current_season: str
    data_co_uk_path: str
    #bookmaker_url: str
    #bookmaker_dictionary: dict
    data_co_uk_url: str
    data_co_uk_dictionary: dict
    fixture_download_url: str
    fixture_download_dictionary: dict
    voting_dict: dict
    matchdays_to_drop: int