import numpy as np
import sys
sys.path.append('.')

import model
import data

class Controller:

    """
    Controller library class.
    """

    _instance = None

    def __new__(self, *args, **kwargs):

        """
        Singleton pattern implementation.
        """

        if not self._instance:

            self._instance = super(Controller, self).__new__(self, *args, **kwargs)

        return self._instance

    def __init__(self):

        """
        Initialize the Controller with text and code classifiers.
        """

        self.text_classifier = model.TextClassifier()
        self.code_classifier = model.CodeClassifier()

        self.text_classifier.load()
        self.code_classifier.load()

    def process(self, text):

        """
        Process the input text to extract and classify code segments.

        Parameters:
            text - input text containing potential code segments.
        """

        segments = []

        for segment in data.extract(text):

            if self.text_classifier.topics[np.argmax(self.text_classifier.predict(segment))] == 'code':

                segments.append(segment)

        for segment in segments:

            label = self.code_classifier.topics[np.argmax(self.code_classifier.predict(segment))]

            print(f"Code segment, {label}\n{segment}\n\n")

