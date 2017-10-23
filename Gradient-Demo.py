import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# helper for variable defs
def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


# enable plot update
plt.ion()
plt.legend()

# Funzione da minimizzare
# y = 0.1 * x + 0.3

num_points = 1000

# STEP 1 - input data build

vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


# STEP 2 - create input and placeholders

# The objective is to generate a TensorFlow code that allows to find the best parameters W and b,
# that from input data x_data, adjunct them to y_data output data, in our case it will be a straight
# line defined by y_data = W * x_data + b .
# The reader knows that W should be close to 0.1 and b to 0.3,
# but TensorFlow does not know and it must realize it for itself.


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

#STEP 3 - model

y = W * x_data + b



#STEP 4 - cost function
#  mean squared error helps us to move forward step by step.
cost = tf.reduce_mean(tf.square(y - y_data))
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
# tf.scalar_summary("cost", cost)


# Initializing the variables
init = tf.global_variables_initializer()


#Step 9 Create a session - START
with tf.Session() as sess:

# The algorithm begins with the initial values of a set of parameters (in our case W and b),
# and then the algorithm is iteratively adjusting the value of those variables in a way that,
# in the end of the process, the values of the variables minimize the cost function.

    sess.run(init)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)


    for step in range(10):
        t1,w1, b1, c1 = sess.run([train_op, W, b, cost])
        # print(step, sess.run(W), sess.run(b), sess.run(cost))
        print(w1, c1)



        ax1.plot(x_data, y_data, 'ro')
        # ax1.plot(x_data, sess.run(W) * x_data + sess.run(b))
        ax1.plot(x_data, w1 * x_data + b1)
        ax2.plot(x_data, w1 * x_data + b1)

        ax1.set_xlabel('x step: {0} cost: {1}'.format(step,c1))
        # plt.xlim(-2, 2)

        # plt.ylim(0.1, 0.6)
        # plt.ylabel('y {0} '.format(step))
        plt.pause(0.5)

        plt.legend()
        plt.show()
