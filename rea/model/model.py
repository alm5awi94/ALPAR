import logging
import os
from typing import List, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from matplotlib import pyplot as plt

from data.data import Data
from processing_module import ProcessingModule

logger = logging.getLogger(__name__)

ShapeType = Union[List[int], Tuple[int]]


class Model(ProcessingModule):
    """
    Module for creating and training feed forward and convolutional
    neural networks.
    """

    def __init__(
            self,
            output_path: str,
            data_path: str,
            nwtype: str,
            hidden_layer_units: List[int],
            hidden_layer_activations: List[str],
            conv_layer_kernels: List[List[int]] = None,
            batch_size: int = 10,
            epochs: int = 100,
            use_class_weights: bool = False,
            val_split: float = 0.1,
            dropout: float = 0.5,
            learning_rate: float = 0.001,
            use_decay: bool = False,
            seed: int = 42,
    ):
        """
        Train an artificial neural network model for a fixed number of epochs.

        :param output_path: path where to save the model
        :param data_path: path to the `Data` output folder
        :param nwtype: The type of network to use ("ff" or "conv")
        :param hidden_layer_units: the number of units for each hidden layer
        :param batch_size: number of samples per gradient update
        :param epochs: number of epochs to train the model
        :param use_class_weights: flag that enables or disables precomputed
         class_weights
         :param dropout: Rate for keras `Dropout` layer
        :param learning_rate: Learning rate for adam optimizer or
            initial learning rate for exponential decay
        :param use_decay: Use adam with exponential decay
        :param seed: random seed for numpy and tensorflow

        """
        super().__init__(output_path, data_path, seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.type = nwtype
        self.hidden_layer_units = hidden_layer_units
        self.hidden_layer_activations = hidden_layer_activations
        if type == "conv" and conv_layer_kernels is None:
            raise ValueError("Type of network is conv, but "
                             "conv_layer_kernels are not provided.")
        self.conv_layer_kernels = conv_layer_kernels
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_class_weights = use_class_weights
        self.val_split = val_split
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_decay = use_decay

    @staticmethod
    def compile_model(model: keras.Model, learning_rate: float = 0.001,
                      learning_decay: bool = False) -> keras.Model:
        """
        Configures the model for training

        :param model: The tensorflow model to compile
        :param learning_rate: Learning rate for adam optimizer or
            initial learning rate for exponential decay
        :param learning_decay: Use adam with exponential decay

        :return: compiled model

        """
        opt: Adam
        if learning_decay:
            # use adam with exponential decay
            lr = ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_rate=0.99,
                decay_steps=50
            )
            opt = Adam(learning_rate=lr)
        else:
            # use constant learning rate
            opt = Adam(learning_rate=learning_rate)
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt, metrics=["accuracy"])
        return model

    @staticmethod
    def create_ff_model(input_shape: ShapeType, output_units: int,
                        units_per_layer: List[int],
                        hidden_layer_activations: List[str],
                        dropout: float = 0.5,
                        learning_rate: float = 0.001,
                        use_decay: bool = False) -> tf.keras.Model:
        """
        Create a feed-forward style neural network with the specified number of
        hidden layers and activations.

        :param input_shape: the shape of the input data
        :param output_units: the number of output units (e.g. classes)
        :param units_per_layer: number of hidden units for each hidden layer
        :param hidden_layer_activations: name of activation function for each
            hidden layer
        :param dropout: Rate for keras `Dropout` layer
        :param learning_rate: Learning rate for adam optimizer or
            initial learning rate for exponential decay
        :param use_decay: Use adam with exponential decay

        :return: A keras Model with a feed-forward structure.

        """
        # create input layer
        input_layer = Input(tuple(input_shape))
        # add hidden layers
        current_layer = input_layer
        # add dense (fully connected) layers in between according to parameters
        for n_units, act in zip(units_per_layer, hidden_layer_activations):
            current_layer = Dense(n_units, activation=act)(current_layer)
        # add dropout to prevent overfitting
        drop = Dropout(dropout)(current_layer)
        # add output layer
        output_layer = Dense(output_units, activation="softmax")(drop)
        # assemble and compile model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return Model.compile_model(model, learning_rate,
                                   learning_decay=use_decay)

    @staticmethod
    def create_conv_model(input_shape: ShapeType,
                          output_units: int,
                          convolutions: List[int],
                          hidden_layer_activations: List[str],
                          conv_layer_kernels: List[List[int]],
                          dropout: float = 0.5,
                          learning_rate: float = 0.001,
                          use_decay: bool = False
                          ) -> tf.keras.Model:
        """
        Create a convolutional network with the specified convolutions and
        input shape.

        :param input_shape: The shape of the input
        :param output_units: The number of output units (e.g. classes)
        :param convolutions: The number of convolutions for each hidden layer
        :param hidden_layer_activations: Activation functions for each hidden
            layer
        :param conv_layer_kernels: Kernel-size for each convolutional layer
        :param dropout: Rate for keras `Dropout` layer
        :param learning_rate: Learning rate for adam optimizer or
            initial learning rate for exponential decay
        :param use_decay: Use adam with exponential decay

        :return: A keras Model with a convolutional structure.

        """
        # create the input layer of the specified shape
        inp = Input(tuple(input_shape))
        h_curr, h_prev = None, None
        # create convolutional layers according to parameters
        for idx, (conv, act, kernel) in enumerate(
                zip(convolutions, hidden_layer_activations, conv_layer_kernels)
        ):
            h_curr = Conv2D(conv, kernel_size=tuple(kernel), activation=act)
            # connect the first convolution to the input
            if idx == 0:
                h_curr = h_curr(inp)
            else:
                h_curr = h_curr(MaxPool2D((2, 2))(h_prev))
            h_prev = h_curr
        # create Flatten layer to map multidimensional filters to the flat
        # output layer
        flat = Flatten()(MaxPool2D((2, 2))(h_curr))
        # add dropout to prevent overfitting
        drop = Dropout(dropout)(flat)
        # create a dense output layer mapping the flattened inputs to classes
        out = Dense(output_units, activation="softmax")(drop)
        # build and compile the model
        model = tf.keras.Model(inputs=inp, outputs=out)
        return Model.compile_model(model, learning_rate,
                                   learning_decay=use_decay)

    def run(self, data: Data = None) -> None:
        """
        Execute the primary module function.

        :param data: (Optional) Can be provided when using the API mode.

        """
        self.setup_data(data)

        input_units: Tuple[int] = (self.data.num_features,)
        output_units: int = self.data.num_classes

        # use different model construction based on the type
        if self.type == "ff":
            if len(self.data.original_shape) > 2:
                raise ValueError("Cannot train ff model with input shape > 2.")
            model: tf.keras.Model = self.create_ff_model(
                input_units,
                output_units,
                self.hidden_layer_units,
                self.hidden_layer_activations,
                self.dropout,
                self.learning_rate,
                self.use_decay
            )
        elif self.type == "conv":
            model: tf.keras.Model = self.create_conv_model(
                # start at 1, because first dimension is always -1 for reshape
                self.data.original_shape[1:],
                output_units,
                self.hidden_layer_units,
                self.hidden_layer_activations,
                self.conv_layer_kernels,
                self.dropout,
                self.learning_rate,
                self.use_decay
            )
        else:
            raise ValueError(f"Unsupported model type '{self.type}'.")

        class_weights = self.data.class_weights
        if not self.use_class_weights:
            class_weights = None

        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.output_dir,
            save_weights_only=False,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        )

        logger.debug(model.summary())

        history = model.fit(
            x=np.reshape(self.data.x_train, self.data.original_shape),
            y=self.data.y_train,
            class_weight=class_weights,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            validation_split=self.val_split,
            callbacks=[model_checkpoint_callback]
        )

        # model.save(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

        # plot training history
        fig, ax_loss = plt.subplots()
        ax_loss.plot(history.history["loss"], label="loss", color="tab:blue")
        ax_loss.plot(history.history["val_loss"], label="val loss",
                     color="tab:orange")
        ax_loss.set_title("Training History")
        ax_loss.set_ylabel("loss")
        ax_loss.set_xlabel("epoch")
        ax_acc = ax_loss.twinx()
        ax_acc.plot(history.history["accuracy"], label="accuracy",
                    color="tab:green")
        ax_acc.plot(history.history["val_accuracy"], label="val accuracy",
                    color="tab:red")
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_xlabel("epoch")
        ax_acc.legend(loc="upper right")
        ax_loss.legend(loc="upper left")
        fig.savefig(os.path.join(self.output_dir, "history.png"))
