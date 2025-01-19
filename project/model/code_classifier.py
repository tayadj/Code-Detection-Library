from model.base import Base

from collections import Counter
import pandas as pd
import numpy as np
import warnings
import string
import math
import re

warnings.filterwarnings("ignore")



class CodeClassifier(Base):

    def __init__(self, hidden_dimension=64, stops_coefficient=0.8, data_path='./data/source/code.csv'):

        super().__init__(hidden_dimension, stops_coefficient, data_path)

    def learn(self, epochs=100, learning_rate=0.001, ngram_size=5):

        for epoch in range (epochs):

            loss = 0
            correct = 0
            quantity = 0

            for value, label in zip(self.data[['Code']].values.flatten(), self.data[['Class']].values.flatten()):

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

    def save(self, path='./model/source/code_classifier.npz'):

        super().save(path)

    def load(self, path='./model/source/code_classifier.npz'):

        super().load(path)
