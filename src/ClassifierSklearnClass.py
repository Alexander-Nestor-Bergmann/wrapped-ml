#!/usr/bin/env python
# encoding: utf-8
from abc import ABC

import numpy as np

from src.BaseSklearnClass import BaseSklearnClass


class ClassifierSklearnClass(BaseSklearnClass, ABC):
    """

    """

    def __init__(self, **base_kwargs):
        """

        :param base_kwargs:
        """

        super().__init__(**base_kwargs)

    def evaluate_on_test_data(self, x_test: np.array, y_test: np.array,
                              return_probability: bool = True, testing_args: dict = None):
        """

        :param x_test:
        :param y_test:
        :param return_probability:
        :param testing_args:
        :return:
        """
        testing_args = {} if testing_args is None else testing_args

        if return_probability:
            predictions = self.model.predict_proba(x_test)[:, 1]
        else:
            predictions = self.model.predict(x_test)

        score = super().evaluate_classifier_on_test_data(y_predicted=predictions, y_true=y_test, **testing_args)

        return score
