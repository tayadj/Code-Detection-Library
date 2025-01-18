from model.base import Base

from collections import Counter
import pandas as pd
import numpy as np
import warnings
import string
import math
import re

warnings.filterwarnings("ignore")



class TextClassifier(Base):

    def __init__(self, hidden_dimension=64, stops_coefficient=0.8, data_path='./data/source/text.csv'):

        super().__init__(data_path, hidden_dimension, stops_coefficient)

    def normalize_text(self, text):
        
        text = text.lower()
        text = re.sub(r'(http|www|https)\S+', '', text)
        text = re.sub(r'([{}])'.format(re.escape(string.punctuation)), r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([word for word in text.split() if word not in self.stops])        
    
        return text

    def convert_text(self, text):
        
        text = self.normalize_text(text)
        inputs = []

        for word in text.split():

            value = np.zeros((self.vocabulary_size, 1))

            try:

                value[self.word_to_index[word]] = 1

            except KeyError:

                pass

            inputs.append(value)

        return inputs

    def load_data(self, path):

        self.data_raw = pd.read_csv(path)
        self.data = self.data_raw.map(self.normalize_text)

    def load_vocabulary(self):

        self.vocabulary = sorted(list(set([word for text in self.data.values.flatten() for word in text.split()])))
        self.vocabulary_size = len(self.vocabulary)

        self.topics = sorted(list(set([word for text in self.data.values.flatten()[1::2] for word in text.split()])))
        self.topics_size = len(self.topics)

        word_counts = Counter(word for text in self.data.values.flatten()[0::2] for word in set(text.split()))
        total_texts = len(self.data.values.flatten()[0::2])
        self.stops = [word for word, count in word_counts.items() if count / total_texts > self.stops_coefficient]

        self.word_to_index = { word : index for index, word in enumerate(self.vocabulary) }
        self.index_to_word = { index : word for index, word in enumerate(self.vocabulary) }

    def load_network(self, dimension_hidden):
        
        self.dimension_input = self.vocabulary_size
        self.dimension_output = self.topics_size
        self.dimension_hidden = dimension_hidden

        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (self.dimension_hidden, self.dimension_input))
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (self.dimension_output, self.dimension_hidden))
        self.weights_hidden_to_hidden = np.random.uniform(-0.5, 0.5, (self.dimension_hidden, self.dimension_hidden))
        self.bias_hidden = np.zeros((self.dimension_hidden, 1))
        self.bias_output = np.zeros((self.dimension_output, 1))

    def forward(self, inputs):

        hidden = np.zeros((self.weights_hidden_to_hidden.shape[0], 1))

        self.recent_inputs = inputs
        self.recent_hidden = { 0: hidden }

        for index, vector in enumerate(inputs):

            hidden = np.tanh(self.weights_input_to_hidden @ vector + self.weights_hidden_to_hidden @ hidden + self.bias_hidden)
            
            self.recent_hidden[index + 1] = hidden

        output = self.weights_hidden_to_output @ hidden + self.bias_output

        return hidden, output

    def backward(self, gradient_output, learning_rate):

        size = len(self.recent_inputs)

        gradient_weights_hidden_to_output = gradient_output @ self.recent_hidden[size].T
        gradient_bias_output = gradient_output

        gradient_weights_hidden_to_hidden = np.zeros(self.weights_hidden_to_hidden.shape)
        gradient_weights_input_to_hidden = np.zeros(self.weights_input_to_hidden.shape)
        gradient_bias_hidden = np.zeros(self.bias_hidden.shape)

        gradient_hidden = self.weights_hidden_to_output.T @ gradient_output

        for time in reversed(range(size)):

            delta = self.activation_derivative(self.recent_hidden[time + 1]) * gradient_hidden
            gradient_bias_hidden += delta
            gradient_weights_hidden_to_hidden += delta @ self.recent_hidden[time].T
            gradient_weights_input_to_hidden += delta @ self.recent_inputs[time].T
            gradient_hidden = self.weights_hidden_to_hidden @ delta

        for gradient in [gradient_weights_input_to_hidden, gradient_weights_hidden_to_hidden, gradient_weights_hidden_to_output, gradient_bias_hidden, gradient_bias_output]:
            
            np.clip(gradient, -1, 1, out = gradient)

        self.weights_hidden_to_output -= learning_rate * gradient_weights_hidden_to_output
        self.weights_hidden_to_hidden -= learning_rate * gradient_weights_hidden_to_hidden
        self.weights_input_to_hidden -= learning_rate * gradient_weights_input_to_hidden
        self.bias_output -= learning_rate * gradient_bias_output
        self.bias_hidden -= learning_rate * gradient_bias_hidden 

    def learn(self, epochs=100, learning_rate=0.001, ngram_size=5):

        for epoch in range (epochs):

            loss = 0
            correct = 0
            quantity = 0

            for value, label in zip(self.data[['Text']].values.flatten(), self.data[['Class']].values.flatten()):

                for value_ in [ngram for n in range(1, min(len(value), ngram_size) + 1) for ngram in self.generate_ngrams(value, n)]:

                    quantity += 1
                    inputs = self.convert_text(value_)
                
                    target = self.topics.index(label)            

                    hidden, output = self.forward(inputs)

                    probabilities = self.activation(output)
            
                    loss += self.loss(probabilities[target])[0]
                    correct += int(np.argmax(probabilities) == target)

                    probabilities[target] -= 1
                    self.backward(probabilities, learning_rate)

            coefficient_loss = loss / quantity
            coefficient_accuracy = correct / quantity

            if epoch % (epochs / 10) == (epochs / 10 - 1):

                print(f'Epoch {epoch+1}:')
                print(f'Loss: {round(coefficient_loss, 3)}')
                print(f'Accuracy: {round(coefficient_accuracy * 100, 3)}%')

    def predict(self, value):

        inputs = self.convert_text(value)
        hidden, output = self.forward(inputs)
        probabilities = self.activation(output)

        return probabilities

    def save(self, path='./model/source/text_classifier.npz'):

        super().save(path)

    def load(self, path='./model/source/text_classifier.npz'):

        super().load(path)
