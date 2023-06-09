import os
import sys
import pickle
import numpy as np 
import pandas as pd
import json
#from sklearn import metrics
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score     # parameters

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):    # save pickle file
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(pd.DataFrame(X_train),y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(pd.DataFrame(X_test))

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path): # to load pickle file
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

def connect_database():
    file_path= os.path.join(os.getcwd(),'mushrooms_token_datastax_cassandra.json')
    with open(file_path) as f:
        file= json.load(f)
        id= file["clientId"]
        secret= file["secret"]

    try: 
        cloud_config= {
        'secure_connect_bundle': os.path.join(os.getcwd(), 'secure-connect-mushrooms.zip')
                    }
        auth_provider = PlainTextAuthProvider(id, secret)
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        session= cluster.connect('ineuron_mlprojects')

        data = session.execute("SELECT * FROM mushroom_csv;")
        return data
    except Exception as e:
        logging.info('Exception Occured in connect_database function utils')
        raise CustomException(e,sys)
