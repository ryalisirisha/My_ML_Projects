import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            num_features=['reading_score','writing_score']
            cat_features=['gender','race/ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline=Pipeline(
                steps= [
                ('imputer',SimpleImputer(Strategy='median')),
                ('scaler', StandardScaler()) 
                ]

            )

            cat_pipeline=Pipeline(
                steps= [
                ('imputer',SimpleImputer(Strategy='median')),
                ('onehotencoder',OneHotEncoder()),
                ('scaler', StandardScaler())
                ]

            )

            logging.info('Numerical and Categorical pipeline created')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,num_features),
                    ('cat_pipelinr',cat_pipeline,cat_features)
                ]
            )

            logging.info('Preprocessor Object Created')

        except Exception as e:
            raise CustomException(e,sys)
        

        return preprocessor
    

    def initiate_data_transformation(self,train_path,test_path)
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            input_features_train_data=train_df.drop(columns=['math_score'],axis=1)
            target_feature_train_data=train_df['math_score']

            input_features_test_data=test_df.drop(columns=['math_score'],axis=1)
            target_feature_test_data=test_df['math_score']
            logging.info('Read train and test data completed')

            preprocessing_obj=self.get_data_transformer_object()
            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_data)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_data)
            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr = np.c_[input_features_train_arr,np.array(target_feature_train_data)]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_data)]

            logging.info('concatenated the input and target features for train and test data')

            logging.info('Save Preprocessing Object')

            save_object(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)






