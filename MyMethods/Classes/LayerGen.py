import numpy as np
from Classes.Layer import Layer


class LayerGenerator:
    def __init__(self,
                 sizes_of_input_layer: int,
                 sizes_of_hidden_layers: iter,
                 sizes_of_output_layer: int,
                 activation_functions: iter,
                 dactivation_functions: iter,
                 function_of_generation: callable) -> None:
        self.sizes_of_input_layer = sizes_of_input_layer
        self.sizes_of_output_layer = sizes_of_output_layer
        self.sizes_of_hidden_layers = sizes_of_hidden_layers
        self.function_of_generation = function_of_generation
        self.activation_functions = activation_functions
        self.dactivation_functions = dactivation_functions

    
    def generate(self):
        return [Layer(self.function_of_generation(i, j), 
                      self.function_of_generation(1, i), 
                      activation_function=af,
                      dactivation_function=daf) for i, j, af, daf in zip([*self.sizes_of_hidden_layers, self.sizes_of_output_layer],
                       [self.sizes_of_input_layer, 
                        *self.sizes_of_hidden_layers], 
                        self.activation_functions, 
                        self.dactivation_functions,)]
        