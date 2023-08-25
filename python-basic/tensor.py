import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
a = tf.constant(1.2)
b = tf.constant(2.3)
print(a+b)
