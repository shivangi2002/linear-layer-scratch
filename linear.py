from random import random

def loss(output, target):
    return (output - target) ** 2

class Linear:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = [random() for _ in range(self.input_size)]
        self.bias = 0
        self.inputs = None
        self.output = None
        self.prev_error = None
        self.lr = 0.01
        

    def forward(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        self.inputs = inputs
        output = self.bias
        for w,i in zip(self.weights, inputs):
           
            output += i * w
        self.output = output
        return output
    
    def backward(self, target):
        if self.output is None:
            raise ValueError("Must call forward() before backward()")
        
        error = (self.output - target)
        
        if self.prev_error is not None:
            if abs(error) > abs(self.prev_error):
                self.lr *= 0.99
                
                
            else:
                self.lr *= 1.01
                
        self.lr = min(self.lr, 0.1)
        self.lr = max(self.lr, 0.0001)
        self.bias -= error * self.lr
    
        for i in range(self.input_size):
            self.weights[i] -= error * self.inputs[i] * self.lr
            
        self.prev_error = error
        