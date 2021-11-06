#!/usr/bin/env python
# encoding: utf-8

from abc import ABCMeta, abstractmethod
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from data.data_paths import SKLEARN_DATAPATH
from src.BaseMachineLearningClass import BaseMachineLearningClass

MODEL_NAME_DICT: dict = {'SVC': SVC, 'SVR': SVR, 'LogisticRegression': LogisticRegression,
                         'GaussianProcessClassifier': GaussianProcessClassifier,
                         'GaussianNB': GaussianNB, 'DecisionTreeClassifier': DecisionTreeClassifier,
                         'KNeighborsClassifier': KNeighborsClassifier, 'RandomForestClassifier': RandomForestClassifier,
                         'AdaBoostClassifier': AdaBoostClassifier, 'KNeighborsRegressor': KNeighborsRegressor,
                         'GaussianProcessRegressor': GaussianProcessRegressor, 'AdaBoostRegressor': AdaBoostRegressor,
                         'RandomForestRegressor': RandomForestRegressor, 'DecisionTreeRegressor': DecisionTreeRegressor,
                         'XGBClassifier': XGBClassifier, 'XGBRegressor': XGBRegressor}


class BaseSklearnClass(BaseMachineLearningClass, metaclass=ABCMeta):
    """

    """

    @abstractmethod
    def __init__(self, sklearn_model_name: str = None, name: str = 'test_model',
                 model_params: dict = None, model_path: str = None):
        """

        :param sklearn_model_name:
        :param name:
        :param model_params:
        :param model_path:
        """
        model_path = SKLEARN_DATAPATH if model_path is None else model_path
        super().__init__(name=name, model_path=model_path)

        # Model params
        self.model_params = {} if model_params is None else model_params
        # Initialise the model
        self.sklearn_model_name = sklearn_model_name
        self.model = None
        self.training_history = None

    @classmethod
    def create_instance_with_saved_model(cls, **class_default_args):
        """

        :param class_default_args:
        :return:
        """

        class_default_args['name'] = class_default_args.get('name', 'best_model')
        class_default_args['model_path'] = class_default_args.get('model_path', SKLEARN_DATAPATH)

        # Initalise the instance
        predictor = cls(**class_default_args)
        predictor.load_model()

        return predictor

    def build_model(self):
        """
        This construction works for only simple sequential models.  Create a subclass to implement more complex
        structures.
        :return:
        """
        self.model = MODEL_NAME_DICT[self.sklearn_model_name](**self.model_params)

    def train_model(self, x_train: np.array, y_train: np.array, save: bool = True):
        """

        :param x_train:
        :param y_train:
        :param save:
        :return:
        """
        if self.model is None:
            self.build_model()

        self.model.fit(x_train, y_train)

        if save:
            Path(f"{self.model_path}").mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, open(f"{self.model_path}/{self.name}.pkl", 'wb'))

    def load_model(self, name: str = None):
        """

        :param name:
        :return:
        """
        name = self.name if name is None else name
        self.model = joblib.load(f"{self.model_path}/{name}.pkl")

    @abstractmethod
    def evaluate_on_test_data(self, x_test: np.array, y_test: np.array):
        """

        :param x_test:
        :param y_test:
        :return:
        """
        raise NotImplementedError
