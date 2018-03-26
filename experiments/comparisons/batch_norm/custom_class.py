from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class ScaleLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ScaleLayer, self).__init__(**kwargs)
    def call(self, x):
        return (x-0.5)*2
    def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)