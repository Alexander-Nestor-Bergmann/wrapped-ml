#!/usr/bin/env python
# encoding: utf-8
from abc import ABC

import numpy as np

from lib.ml_models.BaseKerasNNClass import BaseKerasNNClass


class RegressorNNClass(BaseKerasNNClass, ABC):
    """

    """

    def __init__(self, **base_kwargs):
        """

        :param loss:
        :param base_kwargs:
        """

        metrics = base_kwargs.get('metrics', None)
        if metrics is None:
            base_kwargs['metrics'] = ['RMSE', 'MAE']

        super().__init__(**base_kwargs)

        # Default loss
        self.loss = 'mean_squared_error' if self.loss is None else self.loss

    def evaluate_on_test_data(self, x_test: np.array, y_test: np.array, testing_args: dict = None):
        """

        :param x_test:
        :param y_test:
        :param testing_args:
        :return:
        """
        testing_args = {} if testing_args is None else testing_args

        predictions = self.model.predict(x_test)

        score = super().evaluate_regressor_on_test_data(y_predicted=predictions, y_true=y_test, **testing_args)

        return score
