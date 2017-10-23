# Display of Activation Functions and softmax
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# helper for variable defs
def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


# Funzione da minimizzare
# y = 0.1 * x + 0.3

num_points = 10

# STEP 1 - input data build

vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

print('x_data:------------------------------------------------')
print(x_data)
print('y_data:------------------------------------------------')
print(y_data)

# STEP 2 - create input and placeholders



x_1 = tf.placeholder(tf.float32, None)

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

#STEP 3 - model
y = W * x_data + b
xAll = tf.maximum(x_data, x_data)

xL = tf.lin_space(-6.0,6.0,1000)
ySigmoid = tf.nn.sigmoid(xL)
yRELU = tf.nn.relu(xL)
yELU = tf.nn.elu(xL)
yTANH = tf.nn.tanh(xL)

#STEP 4 - cost function
#  mean squared error helps us to move forward step by step.
# cost = tf.reduce_mean(tf.square(y - y_data))
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
# tf.scalar_summary("cost", cost)

# Initializing the variables
init = tf.global_variables_initializer()

#Step 9 Create a session - START
with tf.Session() as sess:

    sess.run(init)


    print(xL)

    fig1 = plt.figure()

    # N,M,I row col index

    ax1 = fig1.add_subplot(331)
    ax2 = fig1.add_subplot(332)
    ax3 = fig1.add_subplot(333)
    ax4 = fig1.add_subplot(334)

    yS,yR,yE,yT, xL1 = sess.run([ySigmoid, yRELU, yELU, yTANH, xL], feed_dict={x_1 : xL.eval()})
    # print(step, sess.run(W), sess.run(b), sess.run(cost))

    print(yS)
    print(xL1)

    ax1.plot(xL1, yS, 'r')
    ax1.set_title('lll')
    ax2.plot(xL1, yR, 'r')
    ax3.plot(xL1, yE, 'g')
    ax4.set_title('lll')
    ax4.plot(xL1, yT, 'g')


    # ax1.plot(x_data, sess.run(W) * x_data + sess.run(b))
    # ax1.plot(x_data, w1 * x_data + b1)
    # ax2.plot(x_data, yS)

    # ax1.set_xlabel('x step: {0} cost: {1}'.format(step,c1))
    # plt.pause(5)

    plt.legend()
    plt.show()


