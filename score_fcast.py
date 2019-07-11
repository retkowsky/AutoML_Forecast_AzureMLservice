import pickle
import json
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path(model_name = 'AutoML5229ef26ebest') # this name is model.id of model that we want to deploy
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

timestamp_columns = ['WeekStarting']

def run(rawdata, test_model = None):
    """
    Intended to process 'rawdata' string produced by
    
    {'X': X_test.to_json(), y' : y_test.to_json()}
    
    Don't convert the X payload to numpy.array, use it as pandas.DataFrame
    """
    try:
        # unpack the data frame with timestamp        
        rawobj = json.loads(rawdata)                    # rawobj is now a dict of strings        
        X_pred = pd.read_json(rawobj['X'], convert_dates=False)   # load the pandas DF from a json string
        for col in timestamp_columns:                             # fix timestamps
            X_pred[col] = pd.to_datetime(X_pred[col], unit='ms') 
        
        y_pred = np.array(rawobj['y'])                    # reconstitute numpy array from serialized list
        
        if test_model is None:
            result = model.forecast(X_pred, y_pred)       # use the global model from init function
        else:
            result = test_model.forecast(X_pred, y_pred)  # use the model on which we are testing
        
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    
    # prepare to send over wire as json
    forecast_as_list = result[0].tolist()
    index_as_df = result[1].index.to_frame().reset_index(drop=True)
    
    return json.dumps({"forecast": forecast_as_list,   # return the minimum over the wire: 
                       "index": index_as_df.to_json()  # no forecast and its featurized values
                      })
