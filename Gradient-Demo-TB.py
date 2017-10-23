import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf

# helper for variable defs
def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)



# enable plot update
# plt.ion()
# plt.legend()

# Funzione da minimizzare
# y = 0.1 * x + 0.3

num_points = 1000

# STEP 1 - input data build
with tf.name_scope('input'):
    vectors_set = []
    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    #tf.summary.image('input_x', x_data, 10)
    #tf.summary.image('input_y', y_data, 10)


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
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.square(y - y_data))
    # train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)


#STEP 5 - Summary all variables

# Initializing the variables
init = tf.global_variables_initializer()

tf.summary.scalar("cost", cost)

# Create summaries to visualize weights
for var in tf.trainable_variables():
    print(var.name)
    tf.summary.histogram(var.name, var)

## Summarize all gradients
#for grad, var in grads:
#    tf.summary.histogram(var.name + '/gradient', grad)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

#Step 9 Create a session - START
with tf.Session() as sess:

# The algorithm begins with the initial values of a set of parameters (in our case W and b),
# and then the algorithm is iteratively adjusting the value of those variables in a way that,
# in the end of the process, the values of the variables minimize the cost function.

   # Step 10 create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
   summary_writer = tf.summary.FileWriter("./logs/GRADIENT_logs", sess.graph)  # for 0.8
   # merged = tf.summary.merge_all()
   sess.run(init)

   for step in range(100):
       t, summary, c = sess.run([train_op,merged_summary_op,cost])
       print(c)
       summary_writer.add_summary(summary, c)

       print(step, sess.run(W), sess.run(b), sess.run(cost))

print("Run the command line:\n --> tensorboard --logdir=/tmp/tensorflow_logs nThen open http://0.0.0.0:6006/ into your web browser")


