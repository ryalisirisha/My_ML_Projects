import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artiicats','test.csv')
    raw_data_path: str=os.path.join('artificats','data.csv')

class DataIngestion:
    def __int__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_Data_ingestion():
        logging.info("Entered the data ingestion Process")

        try:
            logging.info('Reading the dataset as Pandas DataFrame')
            df=pd.read_csv('data\StudentsPerformance.csv')
            logging.info('Read the dataset as Pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exists_ok=True)
            df.to_csv(self.ingestion.config.raw_data_path,index=False)

            logging.info('TrainTest Splt Initated')
            train_split,test_split=train_test_split(df,test_size=0.2,random_state=42)
            logging.info('TrainTestSplit is completed')
            train_split.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_split.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Train and Test splits are saved to artifacts folder')

            return(
                self.ingestion.config.train_data_path,
                self.ingestion_config.trast_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

