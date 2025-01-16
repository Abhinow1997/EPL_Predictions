import os
import sqlite3 
import pandas as pd 
from loguru import logger 
import sys
from sqlalchemy import TEXT

class SQLliteHandler:
    def __init__(self, league:str, database: str):
        self.league = league
        self.database = database
    
    def get_data(self, table_names):
        """
        Retrieves data from the specified tables in the SQLite database.

        Args:
            table_names (list): A list of table names to retrieve data from.

        Returns:
            list: A list of dataframes corresponding to the tables.
        """
        if isinstance(table_names, str):
            table_names = [table_names]

        dataframes = []
        try:
            connection = sqlite3.connect(self.database)
            for table_name in table_names:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
                dataframes.append(df)
            connection.close()
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
        return dataframes


    def save_dataframes(self, dataframes: list, table_names: list):
        """
        Saves dataframes to the corresponding tables in the SQLite database.

        Args:
            dataframes (list): A list of dataframes to be saved.
            table_names (list): A list of table names corresponding to the dataframes.
        """
        if isinstance(table_names, str):
            table_names = [table_names]
        if isinstance(dataframes, pd.DataFrame):
            dataframes = [dataframes]

        if len(dataframes) != len(table_names):
            logger.error("Length of dataframe_list must be equal to the length of table_names.")
            sys.exit(1)

        try:
            # Connect to the SQLite database
            os.makedirs(os.path.dirname(self.database), exist_ok=True)
            connection = sqlite3.connect(self.database)
            cursor = connection.cursor()

            for df, table_name in zip(dataframes, table_names):
                try:
                    # Save the DataFrame to the corresponding table in the database
                    df.to_sql(table_name, connection, index=False, if_exists='replace')
                    logger.info(f'Table {table_name} created/updated for {self.league} league.')

                except Exception as e:
                    # Print or log the DataFrame to identify the issue
                    logger.debug(f"DataFrame for {table_name}:\n{df.head()}")
                    logger.debug(f"Error saving DataFrame to table {table_name}: {e}")

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")

        finally:
            connection.commit()
            # Close the database connection
            if connection:
                connection.close()