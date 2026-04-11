from random import random


class Linear:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = [random() for _ in range(self.input_size)]
        self.bias = 0
        self.inputs = None
        self.output = None
        
        

    def forward(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        self.inputs = inputs
        output = self.bias
        for w,i in zip(self.weights, inputs):
           
            output += i * w
        self.output = output
        return output
    
    def backward(self, target,lr):
        if self.output is None:
            raise ValueError("Must call forward() before backward()")
        
        error = self.output - target
        self.bias -= error * lr
    
        for i in range(self.input_size):
            self.weights[i] -= error * self.inputs[i] * lr
        