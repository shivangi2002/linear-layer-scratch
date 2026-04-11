from random import random


class Linear:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = [random() for _ in range(self.input_size)]
        self.bias = 0
        
        

    def forward(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        output = self.bias
        for w,i in zip(self.weights, inputs):
           
            output += i * w
        return output
    
    def backward(self, target,lr,output,inputs):
        
        error = output - target
        self.bias -= error * lr
    
        for i in range(self.input_size):
            self.weights[i] -= error * inputs[i] * lr
        