#!/usr/bin/env python
# encoding: utf-8
from abc import ABC

import numpy as np

from lib.ml_models.BaseKerasNNClass import BaseKerasNNClass


class ClassifierNNClass(BaseKerasNNClass, ABC):
    """

    """

    def __init__(self, **base_kwargs):
        """

        :param loss:
        :param base_kwargs:
        """

        metrics = base_kwargs.get('metrics', None)
        if metrics is None:
            base_kwargs['metrics'] = ['accuracy', 'AUC']

        super().__init__(**base_kwargs)

        # Compiling params
        # Default loss to categorical
        if self.loss is None:
            self.loss = 'binary_crossentropy' if self.default_network_architecture[-1][1]['units'] == 1 \
                else 'categorical_crossentropy'

    def evaluate_on_test_data(self, x_test: np.array, y_test: np.array, testing_args: dict = None,
                              padded_output_dims: bool = False):
        """

        :param x_test:
        :param y_test:
        :param testing_args:
        :param padded_output_dims: If the classification is binary, but output is padded to incorporate more info in
                                   the loss function. If True, then y_test = [true_outcome, ..., ...] and
                                   y_pred = [0, ..., prediction, ..., 0]
        :return:
        """
        testing_args = {} if testing_args is None else testing_args

        predictions = self.model.predict(x_test)

        if padded_output_dims:
            predictions = predictions[:, int(predictions[0].size / 2)]
            y_test = y_test[:, 0]

        score = super().evaluate_classifier_on_test_data(y_predicted=predictions, y_true=y_test, **testing_args)

        return score
