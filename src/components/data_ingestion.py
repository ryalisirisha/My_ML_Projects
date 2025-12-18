import os
import sys

from src.exception import CustomException
from src.logger  import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_Data_ingestion(self):
        logging.info("Entered the data ingestion Process")

        try:
            logging.info('Reading the dataset as Pandas DataFrame')
            df=pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read the dataset as Pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('TrainTest Splt Initated')
            train_split,test_split=train_test_split(df,test_size=0.2,random_state=42)
            logging.info('TrainTestSplit is completed')
            train_split.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_split.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Train and Test splits are saved to artifacts folder')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_Data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr =data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))


