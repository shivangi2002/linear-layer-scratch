from random import random


class Linear:
    def __init__(self, input_size):
        self.weights = [random() for _ in range(input_size)]
        self.bias = 0