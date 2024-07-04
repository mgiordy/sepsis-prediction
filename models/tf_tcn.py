import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

@tf.keras.utils.register_keras_serializable()
class single_headed_convolution(tf.keras.layers.Layer):
    # Initialize all the layers inside the single headed convolution
    def __init__(self, time_samples, kernel_size, num_channels, max_pool, **kwargs):
        super().__init__(**kwargs)
        self.tcn_block = []
        self.time_samples = time_samples
        self.max_pool = max_pool
        self.num_levels = len(num_channels)
        self.kernel_size = kernel_size
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            # Same initialisation as pytorch
            initializer = tf.keras.initializers.HeNormal()
            self.tcn_block.append(tf.keras.layers.Conv1D(num_channels[i], self.kernel_size, strides=1, padding="causal", dilation_rate=dilation_size, kernel_initializer=initializer))
            self.tcn_block.append(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))
            self.tcn_block.append(tf.keras.layers.ReLU())
            if self.max_pool[i] != 0:
                self.tcn_block.append(tf.keras.layers.MaxPooling1D(self.max_pool[i]))
            # TODO: dropout

        self.tcn_block = tf.keras.Sequential(self.tcn_block)

    # get_config() is a method inherited from tf.keras.layers.Layer 
    # that returns a dictionary containing the configuration of the layer. 
    # This is used when saving and loading models in order to preserve the structure of the layer and its parameters.
    def get_config(self):
        config = super().get_config()
        config["tcn_block"] = self.tcn_block
        config["max_pool"] = self.max_pool
        config["num_levels"] = self.num_levels
        config["kernel_size"] = self.kernel_size
        return config

    def call(self, x):
        o = tf.reshape(x, (-1,self.time_samples,1))
        o = self.tcn_block(o)
        return o


@tf.keras.utils.register_keras_serializable()
class multi_headed_convolution(tf.keras.layers.Layer):
    # Initialize the single headed convolution layers
    def __init__(self, vital_signs, time_samples, kernel_size, num_channels, max_pool, **kwargs):
        super().__init__(**kwargs)
        self.vital_signs = vital_signs
        self.time_samples = time_samples
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.max_pool = max_pool
        self.vital_signs_list = []
        
        for _ in range(self.vital_signs):
            self.vital_signs_list.append(single_headed_convolution(self.time_samples, self.kernel_size, self.num_channels, self.max_pool))

    # get_config() is a method inherited from tf.keras.layers.Layer that returns a dictionary containing the configuration of the layer. This is used when saving and loading models in order to preserve the structure of the layer and its parameters.
    def get_config(self):
        config = super().get_config()
        config["vital_signs"] = self.vital_signs
        config["vital_signs_list"] = self.vital_signs_list
        config["kernel_size"] = self.kernel_size
        config["num_channels"] = self.num_channels
        config["max_pool"] = self.max_pool
        return config

    
    def call(self, x):
        tcn_out = []
        for i, tcn_i in enumerate(self.vital_signs_list):
            o = tcn_i(x[:,:,i])
            tcn_out.append(o)
        o = tf.concat(tcn_out, axis=1)
        return o


def get_TCN(vital_signs, time_samples, max_pool, num_channels, kernel_size, dense_layers, output_size):
    #input layer necessary for the model to know the shape of the input which allows the model to know the shape of the weights, biases, and activations at each layer
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(time_samples,vital_signs))])

    model.add(multi_headed_convolution(vital_signs, time_samples, kernel_size, num_channels, max_pool))

    #Flatten the concatenated output of the TCNs so that it can be fed to the dense layers
    model.add(tf.keras.layers.Flatten()) #'channels_first'

    initializer = tf.keras.initializers.HeNormal()
    # Dense block
    for dense_out in dense_layers:
        model.add(tf.keras.layers.Dense(dense_out)) #, kernel_initializer=initializer
        model.add(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.01))

    #output Dense
    model.add(tf.keras.layers.Dense(output_size)) #, kernel_initializer=initializer
    model.add(tf.keras.layers.Activation("sigmoid"))

    
    return model