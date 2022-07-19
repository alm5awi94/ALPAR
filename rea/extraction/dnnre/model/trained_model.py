"""
Represent trained Neural Network model
"""
import os
from typing import Tuple

import pandas as pd
import tensorflow.keras.models as models

from ..dnnre_misc import DataValues, OutputClass


# source code adapted to PEP8 from
# https://github.com/sumaiyah/DNN-RE/blob/master/src/model/model.py

class TrainedModel:
    """
    Represent trained neural network model
    """

    def __init__(self, model_path: str, output_classes: Tuple[OutputClass],
                 train_data: DataValues,
                 test_data: DataValues,
                 activations_path: str, ):
        self.model: models.Model = models.load_model(model_path)
        self.activations_path = activations_path

        # self.col_names = col_names
        self.output_classes = output_classes

        self.rules = set()  # DNF rule for each output class
        self.n_layers = len(self.model.layers)

        self.train_data = train_data
        self.test_data = test_data

        self.__compute_layerwise_activations()

    def __compute_layerwise_activations(self):
        """
        Store sampled activations for each layer in CSV files
        """
        # todo make this method work for func and non func keras models
        # Input features of training data
        data_x = self.train_data.X

        # Sample network at each layer
        for layer_index in range(0, self.n_layers):
            partial_model = models.Model(
                inputs=self.model.inputs,
                outputs=self.model.layers[layer_index].output)

            # if output_shape is a list, use the first element as output_shape
            output_shape = self.model.layers[layer_index].output_shape
            if type(output_shape) == list:
                output_shape = output_shape[0]

            # e.g. h_1_0, h_1_1, ...
            neuron_labels = ['h_' + str(layer_index) + '_' + str(i)
                             for i in range(0, output_shape[1])]

            this_activations_path = \
                self.activations_path + str(layer_index) + '.csv'
            # create file if it doesn't exist
            try:
                os.mkdir(self.activations_path)
                open(this_activations_path, 'r').close()
            except IOError:
                open(this_activations_path, 'w').close()

            activation_values = pd.DataFrame(
                data=partial_model.predict(data_x), columns=neuron_labels)
            activation_values.to_csv(this_activations_path, index=False)

        print('Computed layerwise activations.')

    def get_layer_activations(self, layer_index: int):
        """
        Return activation values given layer index
        """
        filename = self.activations_path + str(layer_index) + '.csv'
        return pd.read_csv(filename)

    def get_layer_activations_of_neuron(self, layer_index: int,
                                        neuron_index: int):
        """
        Return activation values given layer index,
         only return the column for a given neuron index
        """
        filename = self.activations_path + str(layer_index) + '.csv'
        return pd.read_csv(filename)[
            'h_' + str(layer_index) + '_' + str(neuron_index)]

    def set_rules(self, rules):
        self.rules = rules

    def print_rules(self):
        for rule in self.rules:
            print(rule)
