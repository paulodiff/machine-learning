# test input variable and softmax exmaples..

import tensorflow as tf
import matplotlib.pyplot as plt


print("Tensorflow version: " + tf.__version__)
tf.set_random_seed(1)

weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
b_test = tf.Variable(tf.random_uniform([6], minval=0.0, maxval=1.0, dtype=tf.float32), name="eeee")

init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)

with tf.Session() as sess:
#print the product
    sess.run(init)

    # print(weights.eval())
    print("b_test")
    print(b_test.eval())

    b_test_softmax = tf.nn.softmax(b_test)

    #print("b_test_softmax")
    print(b_test_softmax.eval())


    print(tf.argmax(b_test_softmax).eval())


    plt.plot(b_test.eval(),'r', label="test")
    plt.plot(b_test_softmax.eval(),'g',label="softmax")
    plt.ylabel('SoftMax demo ...')

    # Now add the legend with some customizations.
    legend = plt.legend(loc='upper left', shadow=True)
    # plt.show()



    #matrix1 = tf.constant([[3., 3.]])
    #matrix2 = tf.constant([[2.], [2.]])
    #product = tf.matmul(matrix1, matrix2)

    # print(product.eval())

    # print(weights.eval())
    # print(biases.eval())
# close the session to release resources
sess.close()