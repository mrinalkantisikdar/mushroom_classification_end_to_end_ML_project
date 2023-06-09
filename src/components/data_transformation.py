import sys
from dataclasses import dataclass # to create a class variable and not any functionalities

import numpy as np 
from numpy import array
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

from src.exception import CustomException
from src.logger import logging


import os
from src.utils import save_object


# here we will do feature engineering; handling: missing values, outlyers; feature scaling, handling catagorical & numerical features 
# input= train & test data path, output = transformed data, pickle files
@dataclass # to create a class variable and not any functionalities
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl') # give path of pickle files

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def target_encode(self, df):
        try:
            logging.info('target encoding initiated')
            target_map= {
                "p": 0, 
                "e": 1
                }
            df.loc[:,'classs'] = df['classs'].map(target_map)
            return df
        except Exception as e:
            logging.info("Error in target encoding")
            raise CustomException(e,sys)


    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            #Define target column
            #target_col= ['classs']
            # Define class mapping
            #class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}

            # Define categorical columns

            categorical_cols = ['cap_surface', 'bruises', 'gill_spacing', 'gill_size', 'gill_color',
       'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
       'ring_type', 'spore_print_color', 'population', 'habitat'] # keeping only the important columns for our model
          
            
                        
            logging.info('Pipeline Initiated')
            
            '''# target column pipeline
            tar_pipeline=Pipeline(
                steps=[
                ('labelencoder', LabelEncoder)
                # no need to standardize after mapping 0 to 1
                ]

           )'''

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder(handle_unknown='ignore', drop= 'first'))
                #('get_dummies',pd.get_dummies(columns=[categorical_cols], drop_first=True))
                # no need to standardize after get dummies
                ]

            )
            # combine target and catagorical pipeline
            preprocessor=ColumnTransformer([
            #('tar_pipeline', tar_pipeline, target_col),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ], remainder ='passthrough', n_jobs=-1)
            logging.info('Pipeline Completed')
            return preprocessor
            
            

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df_pre = pd.read_csv(train_path)
            test_df_pre = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df_pre.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df_pre.head().to_string()}')

            logging.info('Obtaining preprocessing object')


            #target_map_obj = self.target_encode()
            preprocessing_obj = self.get_data_transformation_object()

            # feature engineering

            # Transforming dataframe using target_encode
            train_df= self.target_encode(train_df_pre)
            test_df= self.target_encode(test_df_pre)
                                   
            
            #features_train = array(train_df.columns)
            #features_test = array(test_df.columns)
            target_column_name = 'classs'
            drop_columns = [target_column_name, 'veil_type', 'cap_shape', 'cap_color', 'odor', 'gill_attachment', 
                            'stalk_shape', 'stalk_color_above_ring', 'stalk_color_below_ring', 'ring_number', 'veil_color']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

                  
                      
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            # convert the train & test df to numpy array for fast calculation
            #train_arr = np.c_[input_feature_train_arr, np.concatenate([np.array(target_feature_train_df)], axis=1)] # np.c is concatenation
            #train_arr = pd.concat([pd.DataFrame(input_feature_train_arr), pd.DataFrame(target_feature_train_df)], axis=1)
            #test_arr = pd.concat([pd.DataFrame(input_feature_test_arr), pd.DataFrame(target_feature_test_df)], axis=1)
            #test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            train_arr_df = pd.concat([pd.DataFrame(input_feature_train_arr), pd.DataFrame(target_feature_train_df).reset_index().drop(['index'], axis=1)], axis=1, ignore_index= True, join= 'inner')
            test_arr_df = pd.concat([pd.DataFrame(input_feature_test_arr), pd.DataFrame(target_feature_test_df).reset_index().drop(['index'], axis=1)], axis=1, ignore_index= True, join= 'inner')
            train_arr= train_arr_df.to_numpy() # converting dataframe to numpy array
            test_arr= test_arr_df.to_numpy()

            #train_arr = np.c_[input_feature_train_arr, np.array(pd.DataFrame(target_feature_train_df).reset_index().drop(['index'], axis=1))]
            #test_arr = np.c_[input_feature_test_arr, np.array(pd.DataFrame(target_feature_test_df).reset_index().drop(['index'], axis=1))]



            # saving the pickle files
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
        


'''
            # training dataframe
            train_df[target_column_name]=train_df[target_column_name].map({"p": 0, "e": 1})
            target_feature_train_df=train_df[target_column_name]# target column with p mapped to 0 and e mapped to 1
            train_df['habitat']=train_df['habitat'].map({'d\n':'d', 'l\n':'l', 'g\n':'g', 'm\n':'m', 'p\n':'p', 'u\n':'u', 'w\n':'w', 'l':'l'}) # handling the discripancy in last column due to new line
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1) # first dropping the target column & other unimportant columns
            for cols in input_feature_train_df.columns:
                input_feature_train_df= pd.get_dummies(input_feature_train_df, columns=[cols], drop_first=True)
            


            # same treatment as before for testing dataframe
            test_df[target_column_name]=test_df[target_column_name].map({"p": 0, "e": 1})
            target_feature_test_df=test_df[target_column_name]
            test_df['habitat']=test_df['habitat'].map({'d\n':'d', 'l\n':'l', 'g\n':'g', 'm\n':'m', 'p\n':'p', 'u\n':'u', 'w\n':'w', 'l':'l'})
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            for cols in input_feature_test_df.columns:
                input_feature_test_df= pd.get_dummies(input_feature_test_df, columns=[cols], drop_first=True)


            categorical_cols= ['cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment','gill_spacing','gill_size','gill_color','stalk_shape',
                'stalk_root','stalk_surface_above_ring','stalk_surface_below_ring','stalk_color_above_ring','stalk_color_below_ring','veil_type',
                'veil_color','ring_number','ring_type','spore_print_color','population','habitat']
            '''