from random import random


class Linear:
    def __init__(self, input_size):
        self.weights = [random() for _ in range(input_size)]
        self.bias = 0
        self.input_size = input_size

    def forward(self, inputs):
        self.inputs = inputs
        output = self.bias
        for w,i in zip(self.weights, inputs):
            output += i * w
        return output
    