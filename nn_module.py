import tensorflow as tf
import numpy as np

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits

def conv(x, filter_shape, bias=True, stride=1, padding="VALID", name="conv2d", reuse=None):
    kw, kh, nin, nout = filter_shape

    stddev = np.sqrt(2.0/(np.sqrt(nin*nout)*kw*kh))
    k_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)
    
    with tf.compat.v1.variable_scope(name):
        x = tf.compat.v1.layers.conv2d(x, filters=nout, kernel_size=(kw, kh), strides=(stride, stride), padding=padding, 
                             use_bias=bias, kernel_initializer=k_initializer)
    return x

def deconv(x, filter_shape, bias=True, stride=1, padding="VALID", name="conv2d_transpose", reuse=None):
    kw, kh, nin, nout = filter_shape

    stddev = np.sqrt(1.0/(np.sqrt(nin*nout)*kw*kh))
    k_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)
    with tf.compat.v1.variable_scope(name):
        x = tf.compat.v1.layers.conv2d_transpose(x, filters=nout, kernel_size=(kw, kh), strides=(stride, stride), padding=padding, 
                                       use_bias=bias, kernel_initializer=k_initializer)
    return x

def instance_norm(input, name='instance_norm'):
    with tf.compat.v1.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.compat.v1.get_variable('scale', [depth], initializer=tf.compat.v1.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.compat.v1.get_variable('offset', [depth], initializer=tf.compat.v1.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x=input, axes=[1,2], keepdims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def l1_loss(x, y):
    return tf.reduce_mean(input_tensor=tf.abs(x - y))

def l2_loss(x, y):
    return tf.reduce_mean(input_tensor=tf.square(x - y))

def lrelu(x, leak=0.01, name='lrelu'):
    return tf.maximum(x, leak*x)
