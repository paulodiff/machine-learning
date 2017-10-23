
# idea https://deeplearnjs.org/demos/complementary-color-prediction/complementary_color_prediction.html
# from https://www.tensorflow.org/get_started/estimator
# regression DNNRegression https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/learn
# Complementary Color Prediction


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

print("Tensorflow version: " + tf.__version__)

# Data sets
C_COLOR_TRAINING = "tf-c-color-training.csv"
C_COLOR_TEST = "tf-c-color-test.csv"


# for TensorBoard
MODEL_DIR = "c:/ai/nn/tmp/model/complementarycolor"


print("START OPERAZIONI")

# If the training and test sets aren't stored locally, download them.
#if not os.path.exists(IRIS_TRAINING):
#    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
#    with open(IRIS_TRAINING, "w") as f:
#        f.write(raw)

#if not os.path.exists(IRIS_TEST):
#    raw = urllib.urlopen(IRIS_TEST_URL).read()
#    with open(IRIS_TEST, "w") as f:
#        f.write(raw)


print('Caricamento dataset, specifica il tipo delle FEATURES ')

  # Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=C_COLOR_TRAINING,
      features_dtype=np.int,
      target_dtype=np.int)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=C_COLOR_TRAINING,
      features_dtype=np.int,
      target_dtype=np.int)

print('Stampa di alcuni valori di TRANING')
print(training_set.data)

print('Stampa di alcuni valori di TEST')
print(test_set.data)


print('Impostazione delle colonne')


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]


print('Inizializzazione del classificatore')
# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          # optimizer default Adagrad
                                          # activation_fn default relu
                                          # dropout default none
                                          model_dir=MODEL_DIR)

# Define the training inputs
print('Definizione della funzione di input per il training')

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

print('Training ....')

# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)


print('Definizione della funzione di input per il test')

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

print('Definizione dell"accuratezza del dato usando l"insieme di test')


# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


print('Definizione di alcuni campioni per la prediction')

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5],
    [5.8, 3.1, 5.0, 1.7],
    [6.4, 2.8, 5.6, 2.2],
    [4.9, 3.1, 1.5, 0.1]
     ], dtype=np.float32)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)

pcl = classifier.predict(input_fn=predict_input_fn)

print('Risultato inferenza')
print(pcl)

pc = list(pcl)

print('Risultato inferenza con list()')
print(pc)


print('Stampa lista ..')

for p in pc:
    print(p["classes"])

print('Risultato inferenza usando Classes ...')
predicted_classes = [p["classes"] for p in pc]

print(predicted_classes)

print("New Samples, Class Predictions:    {}\n".format(predicted_classes))

print('OPERAZIONI COMPLETATE')