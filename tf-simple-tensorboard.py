""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

tensorboard --logdir
One variable 2 writer
https://github.com/tensorflow/tensorflow/issues/7089
"""
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import io

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
print("Tensorflow version: " + tf.__version__)

print(datetime.date.today())
print(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'))

folderName = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')

print(folderName)

# exit(0)

DATA_FILE = './data/simple/fire.csv'
MODEL_FOLDER = './model/simple/' + folderName
SKIP_STEP = 10

print('debug: tensorboard --logdir ', MODEL_FOLDER)


# Step 1: read in data from the .xls file
'''
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1
'''

# reader = csv.reader(open(DATA_FILE, "rb"), delimiter=",")
# x = list(reader)
data = np.loadtxt(open(DATA_FILE, "rb"), delimiter=",", skiprows=1)
print(data)
n_samples = data.shape[0] - 1
print('Num of samples:', n_samples)

# exit(0)

'''
data = pd.read_csv(DATA_FILE)
print(data)
print(data.count)
print(data.shape[0])
n_samples = data.shape[0] - 1
print(n_samples)
'''

def gen_plot(d):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(d[:,0],d[:,1],'ro')
    plt.title("Punti di interesse")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
with tf.name_scope("data_input"):
	X = tf.placeholder(tf.float32, name='X')
	Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
with tf.name_scope("data_input"):
	w = tf.Variable(0.1, name='weights')
	b = tf.Variable(0.1, name='bias')

# Step 4: build model to predict Y
Y_predicted = X * w + b 

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
# loss = utils.huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# definisce variabili per TB
tf.summary.scalar("normal/loss", loss)
tf.summary.scalar("normal/current_w", w)            
tf.summary.scalar("normal/current_b", b)            

plot_buf = gen_plot(data)

image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# Add the batch dimension

image = tf.expand_dims(image, 0)
print(image)

tf.summary.image("Example_images", image)
tf.summary.text('tag1', tf.convert_to_tensor('Tag1: Random Text 1'))

# salva il modello
saver = tf.train.Saver() # defaults to saving all variables

with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer()) 
	
	writer = tf.summary.FileWriter(MODEL_FOLDER, sess.graph)
	summaries = tf.summary.merge_all()
	
	# Step 8: train the model
	for index in range(500): # train the model 100 epochs (gira 50 volte il training sugli stessi dati)
		total_loss = 0
		for x, y in data:
			# Session runs train_op and fetch values of loss
            # print('feed {0}: {1}', x, y)
			summary, _, l = sess.run([summaries, optimizer, loss], feed_dict={X: x, Y:y}) 
			total_loss += l
			# img1 = sess.run(image)
			# print('Loss at step {}: {:5.1f}'.format(index, l))
			# summary.value.add(tag="roc", simple_value = total_loss)

		writer.add_summary(summary, global_step=index)

		#if (index + 1) % SKIP_STEP == 0:
		#	print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
		#	total_loss = 0.0
		#	saver.save(sess, MODEL_FOLDER + '/chk-simple', index)

		print('Epoch: {0}: total_loss/n_samples: {1}'.format(index, total_loss/n_samples))

	# close the writer when you're done using it
	writer.close() 
	
	# Step 9: output the values of w and b
	w, b = sess.run([w, b]) 

# plot the results

print('Stampa risultato finale - plot')

X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()

print('Operazioni Terminate')