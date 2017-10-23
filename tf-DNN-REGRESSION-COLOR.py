# https://www.tensorflow.org/get_started/input_fn
# https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/input_fn/boston.py
"""DNNRegressor with custom input_fn for Complementary Color TEST!!!! dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# SOURCE_COLOR_R,SOURCE_COLOR_G,SOURCE_COLOR_B,COMPLEMENT_COLOR_R,COMPLEMENT_COLOR_G,COMPLEMENT_COLOR_B

COLUMNS = ["SOURCE_COLOR","COMPLEMENT_COLOR"]
FEATURES = ["SOURCE_COLOR"]
# LABELS = ["COMPLEMENT_COLOR_R","COMPLEMENT_COLOR_G","COMPLEMENT_COLOR_B"]
LABELS = ["COMPLEMENT_COLOR_R"]
LABEL = "COMPLEMENT_COLOR"

C_COLOR_TRAINING = "tf-c-color-training.csv"
C_COLOR_TEST = "tf-c-color-test.csv"
C_COLOR_PREDICT = "tf-c-color-prediction.csv"

# for TensorBoard
MODEL_DIR = "c:/ai/nn/tmp/model/dnnccolorregression"

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      #y=pd.DataFrame({k: data_set[k].values for k in LABELS}),
      num_epochs=num_epochs,
      shuffle=shuffle)


def main(unused_argv):
  # Load datasets
  training_set = pd.read_csv(C_COLOR_TRAINING, skipinitialspace=True, skiprows=1, names=COLUMNS)
  test_set = pd.read_csv(C_COLOR_TEST, skipinitialspace=True, skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
  prediction_set = pd.read_csv(C_COLOR_PREDICT, skipinitialspace=True, skiprows=1, names=COLUMNS)

  # Feature cols
  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[30, 20, 10],
                                        # label_dimension=2, # modifica per output multi-dimensionale
                                        model_dir=MODEL_DIR)

  # Train
  # regressor.train(input_fn=get_input_fn(training_set), steps=1000)

  # Evaluate loss over one epoch of test_set.
  # ev = regressor.evaluate( input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False ))
  # loss_score = ev["loss"]
  # print("Loss: {0:f}".format(loss_score))

  # Print out predictions over a slice of prediction_set.
  y = regressor.predict( input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False) )
  # .predict() returns an iterator of dicts; convert to a list and print
  # predictions
 
  # predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  predictions = list(p["predictions"] for p in y)
  print("Predictions: {}".format(str(predictions)))


  print("Predictions_set")
  
  for ps in prediction_set:
    print(ps)

  print("predictions")

  for p in predictions:
    print(p)

if __name__ == "__main__":
  tf.app.run()