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
    def learn(self, epochs=100, learning_rate=0.001, ngram_size=10):
        pass

    @abstractmethod
    def predict(self, value):
        pass

    def save(self, path):

        np.savez(path, 
        weights_input_to_hidden = self.weights_input_to_hidden, 
        weights_hidden_to_output = self.weights_hidden_to_output, 
        weights_hidden_to_hidden = self.weights_hidden_to_hidden, 
        bias_hidden = self.bias_hidden, 
        bias_output = self.bias_output, 
        topics = self.topics,
        vocabulary = self.vocabulary,
        dimension_hidden = [self.dimension_hidden],
        stops = self.stops,
        stops_coefficient = [self.stops_coefficient]
        )

    def load(self, path):

        with np.load(path) as loaded:

            self.weights_input_to_hidden = loaded['weights_input_to_hidden']
            self.weights_hidden_to_output = loaded['weights_hidden_to_output']
            self.weights_hidden_to_hidden = loaded['weights_hidden_to_hidden']
            self.bias_hidden = loaded['bias_hidden']
            self.bias_output = loaded['bias_output']

            self.topics = list(loaded['topics'])
            self.topics_size = len(self.topics)

            self.vocabulary = loaded['vocabulary']
            self.vocabulary_size = len(self.vocabulary)

            self.word_to_index = { word : index for index, word in enumerate(self.vocabulary) }
            self.index_to_word = { index : word for index, word in enumerate(self.vocabulary) }

            self.stops = loaded['stops']
            self.stops_coefficient = loaded['stops_coefficient'][0]

            self.dimension_input = self.vocabulary_size
            self.dimension_output = self.topics_size
            self.dimension_hidden = loaded['dimension_hidden'][0]
