import numpy as np
from activations import *

class NetworkLayer():
    def __init__(self, number_of_inputs, number_of_neurons, activation_functions):
        self.weights = 2 * np.random.random((number_of_inputs, number_of_neurons)) - 1
        self.activations = activation_functions
        self.number_of_neurons = number_of_neurons

        self.functions = {
            1: sigmoid_activation,
            2: tanh_activation,
            3: sin_activation,
            4: gauss_activation,
            5: relu_activation,
            6: softplus_activation,
            7: identity_activation,
            8: clamped_activation,
            9: inv_activation,
            10: log_activation,
            11: exp_activation,
            12: abs_activation,
            13: hat_activation,
            14: square_activation,
            15: cube_activation
        }

        self.derivatives = {
            1: sigmoid_derivative,
            2: tanh_derivative,
            3: sin_derivative,
            4: gauss_derivative,
            5: relu_derivative,
            6: softplus_derivative,
            7: identity_derivative,
            8: clamped_derivative,
            9: inv_derivative,
            10: log_derivative,
            11: exp_derivative,
            12: abs_derivative,
            13: hat_derivative,
            14: square_derivative,
            15: cube_derivative
        }

    def think(self, input):
        output = np.zeros(self.activations.size)
        self.last_inputs = input
        i = 0
        for j in np.nditer(self.activations):
            temp = np.dot(input, self.weights[:,i])
            output[i] = self.functions[self.activations[i]](temp)
            i += 1

        return output

    def print_debug(self):
        print(self.weights)
        print("")

class NeuralNetwork():
    def __init__(self, layers):
        self.d_factors = np.zeros(layers[len(layers) - 1].number_of_neurons)
        self.range = 0.1
        self.layer_dict = {}
        self.total_number_of_neurons = 0
        self.neurons_in_layers = np.zeros(len(layers))
        for i in range(0, len(layers)):
            self.layer_dict[i+1] = layers[i]
            self.total_number_of_neurons += layers[i].number_of_neurons
            self.neurons_in_layers[i] = layers[i].number_of_neurons
        self.activation_decisions = np.zeros(self.total_number_of_neurons)

    def set_d_factors(self):
        self.d_factors = np.zeros(self.total_number_of_neurons)

    def make_decision(self, input):
        count = 0
        for i in np.nditer(self.d_factors):
            if(input > i or np.abs(input - i) <= self.range):
                i = input
                return count
            else:
                count += 1
        return None

    #todo: offset neuronIDs by one
    def change_activation(self, functionID, neuronID, d_weight):
        d_factor = self.d_factors[neuronID]
        if(d_weight >= d_factor):
            return True
        elif(np.abs(d_factor - d_weight) <= self.range):
            return True
        else:
            return False

    # need highest value, pos
    def get_suggestion(self, output):
        max = np.amax(output)
        max_pos = np.argmax(output)

        return max, max_pos

    def run_network(self, input):
        output = input
        for i in range(0, len(self.layer_dict)):
            output = self.layer_dict[i+1].think(output)

        return output

    def get_activation(self, neuronID, error):
        input = np.append(error, neuronID)
        output = self.run_network(input)

        activation = int(np.interp(output,[0,1],[0,14])) + 1
        d_temp = np.interp(output,[0,1],[0,14]) + 1
        diff = np.abs(activation - d_temp)
        
        inv = 0
        if(diff != 0):
            inv = 1 / diff
        elif(diff <= 0.01):
            inv = 100 # maximum possible value given a 0.01 resolution

        d_weight = np.interp(inv,[2,100],[0,1])
        
        return activation, d_weight
        #return int(len(self.layers_dict[1].functions) * output)

    def train(self, training_outputs, training_inputs, network_output, iterations):
        length = len(self.layer_dict)
        for l in range(length, 0, -1):
            layer = self.layer_dict[l]
            print("Inputs: {}".format(layer.last_inputs))
            print("Weight: {}".format(layer.weights))
        for i in range(iterations):
            error = np.linalg.norm(training_outputs - network_output)
            length = len(self.layer_dict)
            for l in range(length, 0, -1):
                layer = self.layer_dict[l]
                if(l == length):
                    output = network_output
                else:
                    output = self.layer_dict[l+1].last_inputs
                j = 0
                for n in np.nditer(layer.activations):
                    print(l,j)
                    if(l != 1):
                        delta = error * self.layer_dict[l-1].derivatives[self.layer_dict[l-1].activations[j]](output)
                    else:
                        delta = error * layer.derivatives[layer.activations[j]](output)
                    print("Delat: {}".format(delta))
                    if(l == length):
                        adjustment = output * (delta)
                    else:
                        adjustment = output.dot(delta)
                    print(adjustment)
                    #print(self.layer_dict[1].weights)
                    layer.weights[:,j] += adjustment
                    j += 1
                if(l == length):
                    error = layer.weights.dot(delta)
                else:
                    error = delta.dot(layer.weights)

    def print_debug(self):
        for i in range(0, len(self.layer_dict)):
            print("Layer: {}".format(i+1))
            self.layer_dict[i+1].print_debug()
    
if __name__ == "__main__":
    np.random.seed(1)

    problem = np.array([3,3,4,1,4,3,5,2])
    target = np.array([3,4,1,5,2])
    guess = np.array([1,2,3,4,5])

    n1_number_of_inputs = problem.size
    
    n1_layer1_number_of_neurons = 5
    n1_layer2_number_of_neurons = target.size

    activations_n1_layer1 = np.ones(n1_layer1_number_of_neurons, dtype=np.int8)
    activations_n1_layer2 = np.ones(n1_layer2_number_of_neurons, dtype=np.int8)
    
    n1_layer1 = NetworkLayer(n1_number_of_inputs, n1_layer1_number_of_neurons, activations_n1_layer1)
    n1_layer2 = NetworkLayer(n1_layer1_number_of_neurons, n1_layer2_number_of_neurons, activations_n1_layer2)

    n1_layers = [n1_layer1, n1_layer2]
    n1 = NeuralNetwork(n1_layers)

    n2_number_of_inputs = target.size + 1

    n2_layer1_number_of_neurons = 5
    n2_layer2_number_of_neurons = 5
    n2_layer3_number_of_neurons = 1

    activations_n2_layer1 = np.ones(n2_layer1_number_of_neurons, dtype=np.int8)
    activations_n2_layer2 = np.ones(n2_layer2_number_of_neurons, dtype=np.int8)
    activations_n2_layer3 = np.ones(n2_layer3_number_of_neurons, dtype=np.int8)

    n2_layer1 = NetworkLayer(n2_number_of_inputs, n2_layer1_number_of_neurons, activations_n2_layer1)
    n2_layer2 = NetworkLayer(n2_layer1_number_of_neurons, n2_layer2_number_of_neurons, activations_n2_layer2)
    n2_layer3 = NetworkLayer(n2_layer2_number_of_neurons, n2_layer3_number_of_neurons, activations_n2_layer3)

    n2_layers = [n2_layer1, n2_layer2, n2_layer3]
    n2 = NeuralNetwork(n2_layers)
    n2.set_d_factors()

    debug = False

    if(debug):
        n1.print_debug()
        print("")
        n2.print_debug()

    while(np.array_equal(target, guess) != True):
        n0 = target - guess

        # update activation functions
        for i in range(0, n1.total_number_of_neurons):
            output_n2, d_weight = n2.get_activation(i, n0)
            n1.activation_decisions[i] = d_weight
            
            if(n2.change_activation(output_n2, i, d_weight)):
                count = 0
                id = i
                for l in np.nditer(n1.neurons_in_layers):
                    if(l > id):
                        n1.layer_dict[int(count + 1)].activations[int(id)] = output_n2
                        n2.d_factors[int(i)] = d_weight
                        break
                    else:
                        id -= l
                        count += 1

        output_n1 = n1.run_network(problem)
        d_weight, d_pos = n1.get_suggestion(output_n1)

        pos = n1.make_decision(d_weight)
        print(guess)
        if(pos != None):
            guess[d_pos], guess[pos] = guess[pos], guess[d_pos]

        print(guess)

        for i in range(0, n1.total_number_of_neurons):
            input_n2_temp = np.append(n0, i)
            n2.train(target, input_n2_temp, n1.activation_decisions[i], 1)

        n1.train(target, problem, guess, 1)
        
        input("Press Enter...")
        
        
