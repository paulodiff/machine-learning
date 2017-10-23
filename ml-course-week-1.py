# https://www.coursera.org/learn/ml-foundations/supplement/RP8te/reading-predicting-house-prices-assignment
# WEEK 1
# Regressione lineare sui dati home price prediction in TensorFlow


'''

 my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
 advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# Data sets
DATA_DIR = "./data/home/"
SOURCE_SET = DATA_DIR + "home.csv"
TRAINING_SET = DATA_DIR +  "home-training.csv"
TEST_SET = DATA_DIR + "home-test.csv"
PREDICTION_SET = DATA_DIR + "home-prediction.csv"

# for TensorBoard
MODEL_DIR = "./model/home"

print('## Start operazioni')

all_data = pd.read_csv(SOURCE_SET)
print(all_data)

a = all_data.iloc[2:4,1]
print(a)

# df = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
df = pd.Series(all_data['sqft_living'], all_data['price'])
plt.figure() 
plt.plot(all_data['sqft_living'],all_data['price'],'ro')
plt.show()

print('## Prepara il training set ed il test set ')

print('Sezione di features')
my_features = all_data[['sqft_living','price']]
print(my_features)

all_data['split'] = np.random.randn(all_data[0], 1)

msk = np.random.rand(len(all_data)) <= 0.7

train = df[msk]
test = df[~msk]

print(train)
print(test)


print('## Normalizza i risultati ')


exit(0)

print('## Costruisce il modello')

import tensorflow as tf
print("Tensorflow version: " + tf.__version__)



print('### Fine Operazioni ###')

