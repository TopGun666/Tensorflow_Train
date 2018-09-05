from __future__ import print_function

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constant: %i" % sess.run(a+b))
    print("Multiplication with constant: %i" % sess.run(a*b))


# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some  operation
add = tf.placeholder(tf.int16)
mul = tf.placeholder(tf.int16)

# Launch the default graph.
with tf.Session() as sess:
    # Run tf.Session() as sess:
    print("Addition with variables: %i " % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))


# Create a Constant op that produces a 1x2 matrix. The op is added as a node to the default graph.
# The value returned by the constructor represents the output of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.], [2.]])

# Create ad Matmul op that takes 'matrix1' and 'matrix2' as input.
# The returned value,'product',represents the result of the matrix multiplication.
product = tf.matmul(matrix1, matrix2)

# The output of the op is returned in 'result' as a numpy 'ndarray' object.
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
    # ==> [[12.]]
