# test con i tensori
# test con la gestione delle immagini
# test input variable and softmax examples..
# test con TensorBoard


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import csv
import io


print("Tensorflow version: " + tf.__version__)
tf.set_random_seed(1)

weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
b_test = tf.Variable(tf.random_uniform([6], minval=0.0, maxval=1.0, dtype=tf.float32), name="eeee")

init = tf.global_variables_initializer()

MODEL_FOLDER = './model/test-tf-test'

# Crea un grafico da una serie di punti
def gen_plot(d):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(d[:,0],d[:,1],'ro')
    plt.title("Punti di interesse")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Crea un grafico da una serie di punti
def read_and_decode(d):
    print('Read and decode')
    plt.figure()
    plt.plot(d[:,0],d[:,1],'ro')
    plt.title("Punti di interesse")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


# Aggiorna il tensore con le immagini
def update_images():
    print('update_images')
    return

print('Start operazioni')

dati = np.random.randint(100, size=(20, 2))

current_image_object = read_and_decode(dati)
# tf.concat([t1, t2], 0)
lista_immagini = tf.concat([current_image_object, read_and_decode(dati)],0)

# definisce variabili per TB
tf.summary.histogram("normal/moving_mean", weights)
tf.summary.histogram("normal/loss", biases)
tf.summary.histogram("normal/loss1", Y, Y_
#tf.summary.scalar("normal/current_w", b_test)            
tf.summary.image("img", current_image_object)
tf.summary.image("img2", lista_immagini)

with tf.Session() as sess:
#print the product

    sess.run([init])
    writer = tf.summary.FileWriter(MODEL_FOLDER, sess.graph)
    writer2 = tf.summary.FileWriter(MODEL_FOLDER, sess.graph)

    summaries = tf.summary.merge_all()

    for i in range(5):
        print('Step:',i)

        rd, summary, li = sess.run([current_image_object, summaries, lista_immagini])

        # print(weights.eval())
        print("b_test:")
        print(b_test.eval())


        lista_immagini = tf.zeros([3, 4], tf.int32)

        b_test_softmax = tf.nn.softmax(b_test)

        print("b_test_softmax:")
        print(b_test_softmax.eval())

        # print(current_image_object.eval())

        writer.add_summary(summary, global_step=i)

sess.close()