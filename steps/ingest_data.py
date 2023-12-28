import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path
    """
    def __init__(self, data_path: str):
        """
        :param data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from the data_path
        """
        logging.info(f"Reading data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path

    Args:
        data_path: path to the data
    Returns:
        df: Input dataframe
    """
    try:
        data = IngestData(data_path)
        df = data.get_data()
        logging.info("Data ingestion completed {}".format(df.columns))
        return df
    except Exception as e:
        logging.error(f"Error in ingesting data: {e}")
        raise e