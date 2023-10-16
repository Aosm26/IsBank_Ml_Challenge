import pandas as pd
import tensorflow as tf
import numpy as np
import math
from sklearn.impute import SimpleImputer
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
data=pd.read_csv("hazir.csv")
data.to_csv("mennu1.csv",index=False)
veri=pd.read_csv("mennu1.csv")
satirnumarasi=0
for saniye in data['menus_1']:
  if (saniye==0.0):
    veri=veri.drop(satirnumarasi)
  satirnumarasi+=1
  print(satirnumarasi)
veri.head()
veri.to_csv("m1.csv",index=False)