import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # when you just need to create a class variable and not any functionalities

from src.components.data_transformation import DataTransformation
from src.utils import connect_database



# input= clean data path, output = train and test data path
## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:      # put all the paths here, directory paths are in the form of string
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()     # just the above created class for specifying the paths

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df = pd.DataFrame([d for d in connect_database()]) # read data from local, mongodb, sql (write all these generic codes in utils)
            df = df.iloc[:-1 , :] # # By using iloc[] to select all rows except the last row
            df["stalk_root"]= df["stalk_root"].replace('?', np.nan)
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) # make directory, if already exists don't worry
            df.to_csv(self.ingestion_config.raw_data_path,index=False) # this file will be created
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42) 
            # EDA is done before hand in notebooks, this is test train split

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)     # create train test data files
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(     # return train data & test data path
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)


'''
if __name__=='__main__':
    obj=DataIngestion()
    train_data_path, test_data_path=obj.initiate_data_ingestion()
    data_transformation= DataTransformation()
    train_arr, test_arr,_= data_transformation.initaite_data_transformation(train_data_path, test_data_path)

'''


