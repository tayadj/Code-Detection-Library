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

    """
    Text Classifier based on Base model class.
    """

    def __init__(self, hidden_dimension=64, stops_coefficient=0.8, data_path='./data/source/text.csv'):

        """
        Initializes the class with the given parameters, default parameters specified for Text Classifier.

        Parameters:
            hidden_dimension - size of the hidden layer.
            stops_coefficient - coefficient for determining stop words.
            data_path - path to .csv file.
        """

        super().__init__(hidden_dimension, stops_coefficient, data_path)

    def learn(self, epochs=100, learning_rate=0.001, ngram_size=5):

        """
        Method for training the network.

        Parameters:
            epochs - number of training epochs.
            learning_rate - learning rate for weight updates.
            ngram_size - size of the n-grams.
        """

        for epoch in range (epochs):

            loss = 0
            correct = 0
            quantity = 0

            for value, label in zip(self.data[['Text']].values.flatten(), self.data[['Class']].values.flatten()):

                for value_ in [ngram for n in range(1, min(len(value), ngram_size) + 1) for ngram in self.generate_ngrams(value, n)] + [value]:

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

    def save(self, path='./model/source/text_classifier.npz'):

        """
        Saves the model to a file, default parameters specified for Text Classifier.

        Parameters:
            path - path to .npz file.
        """

        super().save(path)

    def load(self, path='./model/source/text_classifier.npz'):

        """
        Loads the model from a file, default parameters specified for Text Classifier.

        Parameters:
            path - path to .npz file.
        """

        super().load(path)
