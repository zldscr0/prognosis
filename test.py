#load model
from unittest import result
import joblib
loaded_model = joblib.load('model/model.pkl')

import yaml
import pandas as pd
with open('test.yaml', 'r') as file:
    loaded_dict = yaml.safe_load(file)
print("Loaded dictionary:", loaded_dict)
input_data = pd.DataFrame([loaded_dict])

#result
predictions = loaded_model.predict(input_data)
print(predictions)