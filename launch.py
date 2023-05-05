
import pickle
import numpy as np 
import os
from typing import Any, Union,Dict
import os

def loadmodel(logger):
    """Get the model"""
    logger.info("loading model")
    TRAINED_MODEL_FILEPATH = f"regression.pickle"
    with open(TRAINED_MODEL_FILEPATH , 'rb') as f:
        clfdt = pickle.load(f)
    return clfdt  

def preprocessing(df,logger):
    """ Applies preprocessing techniques to the raw data"""
    logger.info("no preprocessing")
    return False
    
def predict(features,model,logger) -> Dict[str, str]:
    """Predicts the results for the given inputs"""
    logger.info("model prediction")
    prediction = model.predict(features)
    return prediction

          
   