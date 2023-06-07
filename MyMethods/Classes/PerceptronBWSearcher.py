import numpy as np


class PerceptronBWSearcher():
    def __init__(self, layers: iter, 
                 activation_function: callable, 
                 dactivation_function_by_dweights: callable,
                 dactivation_function_by_ddisases: callable,
                 batch_spliter_function: callable,
                 error_function: callable,
                 derror_function: callable,
                 n_steps: int):
        self.activation_function = activation_function
        self.dactivation_function_by_dweights = dactivation_function_by_dweights
        self.dactivation_function_by_ddisases = dactivation_function_by_ddisases
        self.batch_spliter_function = batch_spliter_function
        self.error_function = error_function
        self.derror_function = derror_function
        self.layers = layers
        self.n_steps = n_steps
    
    def search(self, X: np.ndarray, Y: np.array):
        for _ in range(self.n_steps):
            for batch_x, batch_y in self.batch_spliter_function(X, Y):
                caches = list()
                output = np.array(batch_x, copy=True)
                outputs = [output]
                # print('output', output.shape, output.tolist())
                for layer in self.layers:
                    # print('W', layer.get_weights().shape, layer.get_weights().tolist())
                    # print('Dias', layer.get_diases().shape, layer.get_diases().tolist())
                    w_p_x = np.dot(output, layer.get_weights().T)
                    # print('W*X', w_p_x.shape, w_p_x.tolist())
                    w_p_x_a_d = w_p_x + layer.get_diases()
                    # print('W*X+d', w_p_x_a_d.shape, w_p_x_a_d.tolist())
                    output, c = self.activation_function(layer, output)
                    # print('output/X', output.shape, output.tolist())
                    caches.append(c)
                    outputs.append(output)
                error = self.error_function(output, batch_y)
                # print('error', error.shape, error.tolist())
                dAL = self.derror_function(output[0], Y)
                # current_cach = cach[-1]
                # doutputs = []
                # dweightses = list()
                # ddiaseses = list()
                current_cache = caches[-1]
                dA_prev_temp, dW_temp, db_temp = self.dactivation_function_by_dweights(dAL, current_cache, 'sigmoid')
                for cach in caches:


                
                
        