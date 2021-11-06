#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod
from inspect import getmembers, isclass, isfunction

# from tcn import TCN
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.optimizers as keras_optimisers
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

from lib.ml_models.BaseMachineLearningClass import BaseMachineLearningClass

from data.data_paths import NN_DATAPATH


class BaseKerasNNClass(BaseMachineLearningClass, metaclass=ABCMeta):
    """

    """

    @abstractmethod
    def __init__(self, example_input: np.array = None, input_shape: tuple = None, name: str = 'test_model',
                 network_architecture: list = None, optimiser_params: dict = None, callbacks: dict = None,
                 metrics: list = None, loss: str = None, training_params: dict = None, class_weights: dict = None,
                 model_path: str = None, allow_plotting: bool = True):
        """

        :param example_input:
        :param input_shape:
        :param name:
        :param network_architecture:
        :param optimiser_params:
        :param callbacks:
        :param metrics:
        :param training_params:
        :param class_weights: Dict with weights as values and keys are classes. Only for classifiers
        :param model_path:
        :param allow_plotting: Let plots to sent to mpl. Not good if e.g. param optimising.
        """
        assert (input_shape is not None or example_input is not None), "Must give input_shape OR example_input"

        model_path = NN_DATAPATH if model_path is None else model_path

        super().__init__(name=name, model_path=model_path)

        # Input size
        if example_input is not None:
            try:
                example_input = np.array(example_input) if not isinstance(example_input, (np.ndarray, np.generic)) \
                    else example_input
            except ValueError:
                print("example_input should be castable as np.array. Network doesn't know input shapes")
                raise

            self.input_shape = example_input.shape
        else:
            self.input_shape = input_shape  # (which is e.g. (num_features, sequence length) for a time-series

        # Network architecture. Just showing an easy sequential topology example here.
        self.default_network_architecture = [('Dense', {'units': 8, 'activation': 'relu'}),
                                             ('Dropout', {'rate': 0.3}),
                                             ('Dense', {'units': 1, 'activation': 'sigmoid'})] \
            if network_architecture is None else network_architecture

        # Training params
        training_params = {} if training_params is None else training_params
        self.num_epochs = training_params.get('num_epochs', 10)
        self.batch_size = training_params.get('batch_size', 32)  # 2^x
        self.validation_split = training_params.get('validation_split', 0.1)

        # Callbacks
        self.callbacks_dict = {'EarlyStopping': {'monitor': 'val_loss', 'min_delta': 0.001, 'patience': 1000,
                                                 'verbose': True},
                               'ModelCheckpoint': {'name': 'best_model', 'monitor': 'val_loss', 'verbose': True,
                                                   'save_best_only': True}
                               } if callbacks is None else callbacks
        self.callbacks_list = []
        if 'EarlyStopping' in self.callbacks_dict.keys():
            e_stop_data = self.callbacks_dict['EarlyStopping']
            early_stopping = EarlyStopping(**e_stop_data)
            self.callbacks_list.append(early_stopping)
        if 'ModelCheckpoint' in self.callbacks_dict.keys():
            check_data = self.callbacks_dict['ModelCheckpoint']
            model_checkpoint = ModelCheckpoint(f"{self.model_path}/{check_data['name']}", **check_data)
            self.callbacks_list.append(model_checkpoint)

        # Metrics
        self.metrics = ['accuracy'] if metrics is None else metrics
        # Compiling params
        # Default loss to categorical
        self.loss = loss

        # Optimiser
        optimiser_input_dict = {} if optimiser_params is None else optimiser_params
        optimiser_params = {k: v for k, v in optimiser_input_dict.items() if k != 'name'}
        self.optimiser = getattr(keras_optimisers, optimiser_input_dict.get('name', 'Adam'))(**optimiser_params)

        # Class imbalances for classifiers
        self.class_weights = class_weights

        # Initialise the model
        self.model = None
        self.training_history = None

        # A dict to access keras layers with their string names e.g. self.LAYER_DICT['Mask'] returns a keras Masking layer
        self.LAYER_DICT = {layer_type[0]: layer_type[1] for layer_type in getmembers(keras_layers, isclass)}
        self.LAYER_DICT['Input'] = Input
        # self.LAYER_DICT['TCN'] = TCN

        self.allow_plotting = allow_plotting

    @classmethod
    def create_instance_with_saved_model(cls, keras_load_args: dict = None, **class_default_args):
        """

        :param keras_load_args:
        :param class_default_args:
        :return:
        """
        keras_load_args = {'custom_objects': {}} if keras_load_args is None else keras_load_args

        class_default_args['name'] = class_default_args.get('name', 'best_model')
        class_default_args['model_path'] = class_default_args.get('model_path', NN_DATAPATH)

        model = load_model(f"{class_default_args['model_path']}/{class_default_args['name']}", **keras_load_args)
        if type(model.input_shape) == list:
            input_shape = model.input_shape[0]
        else:
            input_shape = model.input_shape
        class_default_args['input_shape'] = input_shape

        # Initalise the instance
        predictor = cls(**class_default_args)
        predictor.model = model

        return predictor

    def build_model(self, compile_now: bool = True):
        """
        This construction works for only simple sequential models.  Create a subclass to implement more complex
        structures.
        :param compile_now:
        :return:
        """

        input_layer = self.LAYER_DICT['Input'](shape=self.input_shape)
        last_layer = input_layer
        for layer in self.default_network_architecture:
            next_layer = self.LAYER_DICT[layer[0]](**layer[1])
            last_layer = next_layer(last_layer)

        self.model = Model(inputs=[input_layer], outputs=[last_layer], name=self.name)

        if compile_now:
            self.model.compile(loss=self.loss, optimizer=self.optimiser, metrics=self.metrics)

    def train_model(self, x_train: np.array, y_train: np.array, val_split: float = 0.1, x_val: np.array = None,
                    y_val: np.array = None, shuffle: bool = True, verbose: int = 2, plot_history: bool = False,
                    save: bool = True):
        """

        :param x_train:
        :param y_train:
        :param val_split:
        :param x_val:
        :param y_val:
        :param shuffle:
        :param verbose:
        :param plot_history:
        :param save:
        :return:
        """

        if self.model is None:
            self.build_model()

        if x_val is None or y_val is None:
            self.training_history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                                                   validation_split=val_split, callbacks=self.callbacks_list,
                                                   class_weight=self.class_weights, shuffle=shuffle, verbose=verbose)
        else:
            self.training_history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                                                   validation_data=(x_val, y_val), callbacks=self.callbacks_list,
                                                   class_weight=self.class_weights, shuffle=shuffle, verbose=verbose)

        # if save:
        #     self.model.save(f"{self.model_path}/{self.name}")

        if plot_history:
            self.plot_training_history()

    def load_model(self, name: str = None, **keras_load_args):
        """

        :param name:
        :param keras_load_args:
        :return:
        """
        name = self.name if name is None else name
        self.model = load_model(f"{self.model_path}/{name}", **keras_load_args)

    @abstractmethod
    def evaluate_on_test_data(self, x_test: np.array, y_test: np.array):
        """

        :param x_test:
        :param y_test:
        :return:
        """
        raise NotImplementedError

    def plot_training_history(self, metrics: list = None, show_now: bool = False):
        """

        :return:
        """
        assert self.training_history is not None, "Train model first!"

        if not self.allow_plotting:
            return

        metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss'] if metrics is None else metrics
        # metrics = self.metrics if metrics is None else metrics
        metrics = [m.lower() if type(m) == str else m.__name__ for m in metrics]

        f, ax = plt.subplots()
        for metric_to_plot in metrics:
            ax.plot(self.training_history.history[metric_to_plot], label=metric_to_plot)

        ax.set_title('Training history')
        ax.set_ylabel('Metric')
        ax.set_xlabel('Epoch')
        ax.legend()

        if show_now:
            plt.show()

    def plot_network_architecture(self, with_shapes: bool = True):
        """

        :param with_shapes:
        :return:
        """
        plot_model(self.model, f"{self.name}.png", show_shapes=with_shapes)

