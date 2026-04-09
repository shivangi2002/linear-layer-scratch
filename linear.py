from random import random


class Linear:
    def __init__(self, inputs):
        self.input_size = len(inputs)
        self.weights = [random() for _ in range(self.input_size)]
        self.bias = 0
        
        self.inputs = inputs
        

    def forward(self):
        output = self.bias
        for w,i in zip(self.weights, self.inputs):
           
            output += i * w
        return output
    
    def backward(self, target,lr):
        output = self.forward()
        error = output - target
        self.bias -= error * lr
    
        for i in range(self.input_size):
            self.weights[i] -= error * self.inputs[i] * lr
            