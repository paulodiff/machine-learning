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

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
print("Tensorflow version: " + tf.__version__)

COLUMNS = ["sqft_living", "price"]
FEATURES = ["sqft_living", "price"]
LABEL = "price"

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

def build_dataset():
    
    print('## Build Data Set')

    all_data = pd.read_csv(SOURCE_SET)
    print(all_data)

    # Visualizzazione dati
    # df = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    '''
    df = pd.Series(all_data['sqft_living'], all_data['price'])
    plt.figure() 
    plt.plot(all_data['sqft_living'],all_data['price'],'ro')
    plt.show()
    '''

    print('Sezione di features')
    my_features = all_data[['sqft_living','price']]
    print(my_features)

    all_data['split'] = np.random.randn(all_data.shape[0], 1)

    msk = np.random.rand(len(all_data)) <= 0.7

    train = all_data[msk]
    test = all_data[~msk]

    print(train)
    print(test)

    print(len(train))
    print(len(test))

    print('Save data :', TRAINING_SET)
    train.to_csv(TRAINING_SET, encoding='utf-8-sig')

    print('Save data :', TEST_SET)
    test.to_csv(TEST_SET, encoding='utf-8-sig')

    return

def main(unused_argv):

    # Cotruisce i file per il modello
    build_dataset()

    print('Loading datasets')

    # Load datasets
    training_set = pd.read_csv(TRAINING_SET, skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv(TEST_SET, skipinitialspace=True, skiprows=1, names=COLUMNS)

    # Set of 6 examples for which to predict median house values
    # prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

    # Feature cols
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    print('Regressor...')

    # Build 2 layer fully connected DNN with 10, 10 units respectively.
    regressor = tf.estimator.LinearRegressor(feature_columns=feature_cols,
                                        # hidden_units=[30, 20, 10],
                                        # label_dimension=2, # modifica per output multi-dimensionale
                                        model_dir=MODEL_DIR)


    print('Training...')
    # Train
    regressor.train(input_fn=get_input_fn(training_set), steps=5000)


    print('Evaluate...')
    # Evaluate loss over one epoch of test_set.
    ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
    loss_score = ev["loss"]
    
    print("Loss: {0:f}".format(loss_score))

    print(ev)

    print('### Fine Operazioni ###')


if __name__ == "__main__":
  tf.app.run()
