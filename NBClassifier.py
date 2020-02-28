from collections import Counter, defaultdict
import numpy as np
from pprint import pprint


class NBClassifier():


    def train(self, X, y):
        self.data = X
        self.labels = np.unique(y)
        self.class_probabilities = self._calculate_relative_proba(y)
        pprint(self.class_probabilities)
        self._initialize_nb_dict()  # Creates the dict which stores the information gathered durin training
        examples_no, features_no = self.data.shape

        for label in self.labels:
            X_class = []  # a list which will contain all the examples for a specific class/label
            for example_index, example_label in enumerate(y):
                if example_label == label:
                    X_class.append(X[example_index])

            examples_class_no, features_class_no = np.shape(X_class)

            for feature_index in range(features_class_no):
                for item in X_class:
                    self.nb_dict[label][feature_index].append(item[feature_index])

        # Now we have a dictionary containing all occurences of feature values, per feature, per class
        # We need to transform this dict to a dict with relative feature value probabilities per class
        for label in self.labels:
            for feature_index in range(features_no):
                self.nb_dict[label][feature_index] = self._calculate_relative_proba(self.nb_dict[label][feature_index])

    def predict(self, X_new):
        Y_dict = {}

        # First we determine the class-probability of each class, and then we determine the class with the highest probability
        for label in self.labels:
            class_probability = self.class_probabilities[label]

            for feature_index in range(len(X_new)):
                relative_feature_values = self.nb_dict[label][feature_index]
                if X_new[feature_index] in relative_feature_values.keys():
                    class_probability *= relative_feature_values[X_new[feature_index]]
                else:
                    class_probability *= 0.01  # Lidstone smoothing
            Y_dict[label] = class_probability

        return self._get_class(Y_dict)

    def _initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)

    @staticmethod
    def _calculate_relative_proba(elements_list):
        no_examples = len(elements_list)
        occurrence_dict = dict(Counter(elements_list))

        for key in occurrence_dict.keys():
            occurrence_dict[key] = occurrence_dict[key] / float(no_examples)

        return occurrence_dict

    @staticmethod
    def _get_class(score_dict):
        sorted_dict = sorted(score_dict.items(), key=lambda value: value[1], reverse=True)
        sorted_dict = dict(sorted_dict)
        keys_list = list(sorted_dict.keys())
        max_key = keys_list[0]
        return max_key
