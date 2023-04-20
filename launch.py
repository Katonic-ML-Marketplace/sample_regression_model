
import pickle
import numpy as np 
import os
from typing import Any, Union,Dict
import os

def loadmodel(logger):
    """Get model from cloud object storage."""
    logger.info("loading model")
    TRAINED_MODEL_FILEPATH = f"regression.pkl"
    with open(TRAINED_MODEL_FILEPATH , 'rb') as f:
        clfdt = pickle.load(f)
    return clfdt  

def preprocessing(df:np.ndarray,logger):
    """ Applies preprocessing techniques to the raw data"""
    logger.info("no preprocessing")
    return False
    
def predict(features: np.ndarray,model:Any,logger) -> Dict[str, str]:
    """Predicts the results for the given inputs"""
    logger.info("model prediction")
    prediction = model.predict(features)
    return prediction

          
   
