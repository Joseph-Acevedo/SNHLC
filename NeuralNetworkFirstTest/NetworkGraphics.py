from graphics import *
import math

class VisualizeNetwork():
    
    def __init__(self):
        print("Starting Graphics")
        self.COLORS = {
            0: color_rgb(0, 255, 0),
            1: color_rgb(34, 255, 0),
            2: color_rgb(68, 255, 0),
            3: color_rgb(102, 255, 0),
            4: color_rgb(171, 255, 0),
            5: color_rgb(205, 255, 0),
            6: color_rgb(239, 255, 0),
            7: color_rgb(255, 239, 0),
            8: color_rgb(255, 205, 0),
            9: color_rgb(255, 171, 0),
            10: color_rgb(255, 137, 0),
            11: color_rgb(255, 102, 0),
            12: color_rgb(255, 68, 0),
            13: color_rgb(255, 34, 0),
            14: color_rgb(255, 0, 0)}
        

    def start_graphics(self, network, net_name):
        print("Starting {}".format(net_name))
        # CONSTANTS
        self.WINDOW_WIDTH  = 800
        self.WINDOW_HEIGHT = 400
        self.X_SAFE_ZONE   = self.WINDOW_WIDTH  * 0.05
        self.Y_SAFE_ZONE   = self.WINDOW_HEIGHT * 0.1
        self.X_SAFE_WIDTH  = self.WINDOW_WIDTH  - 2 * self.X_SAFE_ZONE
        self.Y_SAFE_HEIGHT = self.WINDOW_HEIGHT - 2 * self.Y_SAFE_ZONE
        self.NEURON_SIZE   = 20
        self.WEIGHT_STROKE = 7

        # GLOBAL VARIABLES
        self.RUN_GRAPHICS = true

        # NETWORK CHARACTERISTICS
        self.name = net_name
        self.layers = network.layer_dict
        self.number_of_layers = len(self.layers)
        self.space_between_layers = self.X_SAFE_WIDTH / (self.number_of_layers)

        # GRAPHICS
        self.window = GraphWin(net_name, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.window.setBackground(color_rgb(105,105,105))
        
        self.neurons    = []
        self.actv_funcs = []
        space_between_neurons = 0
        if self.layers[1].number_of_inputs > 1:
            space_between_neurons = self.Y_SAFE_HEIGHT / (self.layers[1].number_of_inputs - 1)
            
        for n in range(0, self.layers[1].number_of_inputs): # Draw input layer
            circle = Circle(Point(self.X_SAFE_ZONE, self.Y_SAFE_ZONE + (space_between_neurons * n)), self.NEURON_SIZE)
            #circle.draw(self.window)
            self.neurons.append(circle)
        
        for lay in range(1, self.number_of_layers + 1): # Draw layers 1 - number_of_layers
            self.setup_layer(self.layers[lay], lay)


        print("Network: {} has {} neurons".format(net_name, len(self.neurons)))
        self.weight_lines = []
        for layer in range(1, self.number_of_layers + 1):
            self.draw_weights(layer)

        for line in self.weight_lines:
            line.draw(self.window)

        for neuron in self.neurons:
            neuron.setFill('white')
            neuron.draw(self.window)

        for func in self.actv_funcs:
            func.setSize(10)
            func.draw(self.window)

        while self.RUN_GRAPHICS:
            mouse = self.window.checkMouse()
            if mouse != None:
                
        #while self.window.checkMouse() == None:
        #self.update(network, net_name)

        #self.update(network, net_name)
        self.window.getMouse()
        self.window.close()


    """
    @param point: The point you want to check, as a tuple
    @param center: The center of the circle you're checking, as a tuple
    @param radius: The radius, in pixels, of the circle
    """
    def point_in_circle(point, center, radius):
        d = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        if d >= radius:
            return True
        else:
            return False
    

    """
    Draws the neurons of a single layer
    @param layer: The layer object to draw
    @param key: The position in the dictionary the layer is
    """
    def setup_layer(self, layer, key):
        #print("Netwok: {}\tLayer {}: \n{}".format(self.name, key, layer.weights))
        space_between_neurons = 0
        if layer.number_of_neurons > 1:
            space_between_neurons = self.Y_SAFE_HEIGHT / (layer.number_of_neurons - 1)

            for n in range(0, layer.number_of_neurons):
                circle = Circle(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + (space_between_neurons * n)), self.NEURON_SIZE)
                actv_func = Text(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + (space_between_neurons * n)), str(layer.activations[n]))
        
                self.neurons.append(circle)
                self.actv_funcs.append(actv_func)
        else:
            circle = Circle(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + self.Y_SAFE_HEIGHT / 2), self.NEURON_SIZE)
            actv_func = Text(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + self.Y_SAFE_HEIGHT / 2), str(layer.activations[0]))

            self.neurons.append(circle)
            self.actv_funcs.append(actv_func)

    """
    Updates the layer
    @param layer: The layer object to update
    @param key: The position in the dictionary the layer is
    @param neurons: The list to append neurons to
    @param actv_funcs: The list of activation funcations to append functions to
    """
    def update_layer(self, layer, key, neurons, actv_funcs):
        space_between_neurons = 0
        if layer.number_of_neurons > 1:
            space_between_neurons = self.Y_SAFE_HEIGHT / (layer.number_of_neurons - 1)

            for n in range(0, layer.number_of_neurons):
                circle = Circle(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + (space_between_neurons * n)), self.NEURON_SIZE)
                actv_func = Text(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + (space_between_neurons * n)), str(layer.activations[n]))
        
                neurons.append(circle)
                actv_funcs.append(actv_func)
        else:
            circle = Circle(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + self.Y_SAFE_HEIGHT / 2), self.NEURON_SIZE)
            actv_func = Text(Point(self.X_SAFE_ZONE + (key * self.space_between_layers), self.Y_SAFE_ZONE + self.Y_SAFE_HEIGHT / 2), str(layer.activations[0]))

            neurons.append(circle)
            actv_funcs.append(actv_func)
        

    """
    Draws lines with thickness corresponding to weights
    @param key: The position in the dictionary the layer is
    """
    def draw_weights(self, key):
        weight_txt = []
        if key == 1:
            # Layer 1, corresponds to input neurons
            weights = self.layers[key].weights
            # layer_beg   = self.layers[key - 1] # We want to draw the lines starting from the previous neurons
            layer   = self.layers[key]
            start_neuron = 0                   # The position in list of neurons that the layer starts

            for i in range(0, layer.number_of_inputs):
                for l in range(0, layer.number_of_neurons ):
                    line = Line(self.neurons[start_neuron + i].getCenter(), self.neurons[start_neuron + layer.number_of_inputs + l].getCenter())
                    line.setWidth(int(((weights[i][l] % 0.15) / 0.05)))
                    #print("Layer {}: W{}{}".format(key, i, l))
                    line.setFill(self.COLORS[abs(int(weights[i][l]/0.15))])
                    self.weight_lines.append(line)
                    
                    
        else:
            weights = self.layers[key].weights
            layer_beg   = self.layers[key - 1] # We want to draw the lines starting from the previous neurons
            layer_end   = self.layers[key]
            start_neuron = self.layers[1].number_of_inputs                   # The position in list of neurons that the layer starts
            for i in range(1, key - 1):
                start_neuron += self.layers[i].number_of_neurons

            for i in range(0, layer_beg.number_of_neurons):
                for l in range(0, layer_end.number_of_neurons ):
                    line = Line(self.neurons[start_neuron + i].getCenter(), self.neurons[start_neuron + layer_beg.number_of_neurons + l].getCenter())
                    line.setWidth(int(((weights[i][l] % 0.15) / 0.05)))
                    #print("Layer {}: W{}{}".format(key, i, l))
                    line.setFill(self.COLORS[abs(int(weights[i][l]/0.15))])
                    self.weight_lines.append(line)


    """
    Draws lines with thickness corresponding to weights
    @param key: The position in the dictionary the layer is
    @param neurons: The list to read and write neurons to
    @param actv_funcs:  The list to read and write activations functions to
    @param weight_conns: The list of Lines to append the connections to
    """
    def update_weights(self, key, neurons, actv_funcs, weight_conns):
        
        if key == 1:
            # Layer 1, corresponds to input neurons
            weights = self.layers[key].weights
            # layer_beg   = self.layers[key - 1] # We want to draw the lines starting from the previous neurons
            layer   = self.layers[key]
            start_neuron = 0                   # The position in list of neurons that the layer starts

            for i in range(0, layer.number_of_inputs):
                for l in range(0, layer.number_of_neurons ):
                    line = Line(neurons[start_neuron + i].getCenter(), neurons[start_neuron + layer.number_of_inputs + l].getCenter())
                    line.setWidth(int(((weights[i][l] % 0.15) / 0.05)))
                    #print("Layer {}: W{}{}".format(key, i, l))
                    line.setFill(self.COLORS[abs(int(weights[i][l]/0.15))])
                    weight_conns.append(line)

                    
        else:
            weights = self.layers[key].weights
            layer_beg   = self.layers[key - 1] # We want to draw the lines starting from the previous neurons
            layer_end   = self.layers[key]
            start_neuron = self.layers[1].number_of_inputs                   # The position in list of neurons that the layer starts
            for i in range(1, key - 1):
                start_neuron += self.layers[i].number_of_neurons

            for i in range(0, layer_beg.number_of_neurons):
                for l in range(0, layer_end.number_of_neurons ):
                    line = Line(neurons[start_neuron + i].getCenter(), neurons[start_neuron + layer_beg.number_of_neurons + l].getCenter())
                    line.setWidth(int(((weights[i][l] % 0.15) / 0.05)))
                    #print("Layer {}: W{}{}".format(key, i, l))
                    line.setFill(self.COLORS[abs(int(weights[i][l]/0.15))])
                    weight_conns.append(line)

    """
    Updates the weights and activation functions of the drawn network
    @param network: The network with updated values
    """
    def update(self, network, net_name):
        
        self.name = net_name
        self.layers = network.layer_dict
        self.number_of_layers = len(self.layers)
            
        temp_neurons      = []
        temp_actv_funcs   = []
        temp_weight_lines = []
        space_between_neurons = 0
        if self.layers[1].number_of_inputs > 1:
            space_between_neurons = self.Y_SAFE_HEIGHT / (self.layers[1].number_of_inputs - 1)
                
        for n in range(0, self.layers[1].number_of_inputs): # Draw input layer
            circle = Circle(Point(self.X_SAFE_ZONE, self.Y_SAFE_ZONE + (space_between_neurons * n)), self.NEURON_SIZE)
            temp_neurons.append(circle)
            
        for lay in range(1, self.number_of_layers + 1): # Draw layers 1 - number_of_layers
             self.update_layer(self.layers[lay], lay, temp_neurons, temp_actv_funcs)

        for layer in range(1, self.number_of_layers + 1):
            self.update_weights(layer, temp_neurons, temp_actv_funcs, temp_weight_lines)

        for idx, line in enumerate(temp_weight_lines):
            self.weight_lines[idx].undraw()
            self.weight_lines[idx] = line
            line.draw(self.window)

        for idx, neuron in enumerate(temp_neurons):
            self.neurons[idx].undraw()
            self.neurons[idx] = neuron
            neuron.setFill('white')
            neuron.draw(self.window)

        for idx, func in enumerate(temp_actv_funcs):
            self.actv_funcs[idx].undraw()
            self.actv_funcs[idx] = func
            func.setSize(10)
            func.draw(self.window)
        
