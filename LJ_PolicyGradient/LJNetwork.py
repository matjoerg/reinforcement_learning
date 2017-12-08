import numpy as np
import tensorflow as tf



def Qnetwork():
    """
    """

    # Qgraph = tf.Graph()
    
    # with Qgraph.as_default():
    CurrentFeature = tf.placeholder(tf.float32,shape=[None,20])
    layer = tf.contrib.layers.fully_connected(inputs=CurrentFeature,
                                              num_outputs = 100)
    CurrentInput = tf.contrib.layers.fully_connected(inputs=layer,
                                                     activation_fn = None,
                                                     num_outputs = 10)

    NextFeature = tf.placeholder(tf.float32,shape=[None,20])
    layer = tf.contrib.layers.fully_connected(inputs=NextFeature,
                                              num_outputs = 100)
    NextInput = tf.contrib.layers.fully_connected(inputs=layer,
                                                  activation_fn = None,                                                  
                                                  num_outputs = 10)

    inputs = tf.concat([CurrentInput,NextInput],axis = 1)
    layer = tf.contrib.layers.fully_connected(inputs = inputs,
                                              num_outputs = 100)
    layer = tf.contrib.layers.fully_connected(inputs = layer,
                                              num_outputs = 100)
    Q = tf.contrib.layers.fully_connected(inputs = layer,
                                          activation_fn = None,
                                          num_outputs = 1)

    return Q,CurrentFeature,NextFeature


