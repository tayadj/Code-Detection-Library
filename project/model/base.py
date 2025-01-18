from abc import ABC, abstractmethod

import numpy as np

class Base(ABC):

    def __init__(self, data_path, hidden_dimension=64, stops_coefficient=0.8):

        self.activation = lambda x: np.exp(x) / sum(np.exp(x))
        self.activation_derivative = lambda x: (1 - x ** 2)
        self.loss = lambda x: -np.log(x)
        
        self.stops_coefficient = stops_coefficient
        self.stops = []
        
        self.load_data(data_path)
        self.load_vocabulary()
        self.load_network(hidden_dimension)

    def generate_ngrams(self, text, n):

        words = text.split()
        ngrams = []

        for i in range(len(words) - n + 1):

            ngrams.append(' '.join(words[i:i + n]))

        return ngrams

    @abstractmethod
    def normalize_text(self, text):
        pass

    @abstractmethod
    def convert_text(self, text):
        pass

    @abstractmethod
    def load_data(self, path):
        pass

    @abstractmethod
    def load_vocabulary(self):
        pass

    @abstractmethod
    def load_network(self, dimension_hidden):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, gradient_output, learning_rate):
        pass

    @abstractmethod
    def learn(self, epochs=100, learning_rate=0.001):
        pass

    @abstractmethod
    def predict(self, value):
        pass

    @abstractmethod
    def save(self, path='./data/model.npz'):
        pass

    @abstractmethod
    def load(self, path='./data/model.npz'):
        pass
