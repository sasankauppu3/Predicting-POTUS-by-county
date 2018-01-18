
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import pickle
from PreProcess import *

SavedModelFile = "model.pkl"
filePath = "test_potus_by_county.csv"
X=pd.read_csv(filePath)
X=PreProcess(X)

try:
    with open(SavedModelFile, 'rb') as f:
        model = pickle.load(f)
except:
    print("Did not find a saved model, please run build_model.py")
    exit()
    
predict = [pred for pred in model.predict(X)]

with open('predictions.csv', 'w+') as f:
    f.write("Winner\n")
    for pred in predict:
        f.write(pred+"\n")

