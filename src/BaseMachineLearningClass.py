#!/usr/bin/env python
# encoding: utf-8

import dill
import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, List


class BaseMachineLearningClass(object, metaclass=ABCMeta):
    """

    """

    @abstractmethod
    def __init__(self, name: str = 'test_model', model_path: str = 'data/models'):
        """

        :param name:
        :param model_path:
        """
        self.model = None
        self.name = name
        self.model_path = model_path

    @classmethod
    def create_instance_with_saved_model(cls):
        """

        :return:
        """
        raise NotImplementedError

    @classmethod
    def load_saved_class_instance(cls, name: str = None, path: str = None):
        """

        :param name:
        :param path:
        :return:
        """
        name = 'best_model' if name is None else name
        path = 'data/models' if path is None else path

        with open(f"{path}/{name}.pkl", 'wb') as s:
            predictor: cls = dill.load(s)

        return predictor

    def save_class_instance(self, name: str = None, path: str = None):
        """

        :param name:
        :param path:
        :return:
        """
        name = self.name if name is None else name
        path = self.model_path if path is None else path

        # Pickle
        with open(f"{path}/{name}.pkl", 'wb') as s:
            dill.dump(self, s)

    @abstractmethod
    def build_model(self):
        """

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def train_model(self, x_train: np.array, y_train: np.array, save: bool = True):
        """

        :param x_train:
        :param y_train:
        :param save:
        :return:
        """

        raise NotImplementedError()

    def calibrate_probabilities(self, x_test: np.array, y_test: np.array, n_kfolds: int = 0,
                                sklearn_args: dict = None):
        """

        :param x_test:
        :param y_test:
        :param n_kfolds:
        :param sklearn_args:
        :return:
        """
        sklearn_args = {} if sklearn_args is None else sklearn_args

        sklearn_args['cv'] = n_kfolds if n_kfolds > 0 else "prefit"

        self.model = CalibratedClassifierCV(self.model, **sklearn_args)
        self.model.fit(x_test, y_test)

    @abstractmethod
    def load_model(self, name: str = None):
        """

        :param name:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_on_test_data(self, x_test: np.array, y_test: np.array):
        """

        :param x_test:
        :param y_test:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def evaluate_classifier_on_test_data(y_predicted: np.array, y_true: np.array, score_to_return: str = None,
                                         print_classification_report: bool = True, plot_roc: bool = True,
                                         plot_confusion_matrix: bool = True, show_now: bool = True) -> float:
        """

        :param y_predicted:
        :param y_true:
        :param score_to_return:
        :param print_classification_report:
        :param plot_roc:
        :param plot_confusion_matrix:
        :param show_now:
        :return:
        """
        score_to_return = 'brier' if score_to_return is None else score_to_return
        assert score_to_return in ['brier', 'roc_auc']
        score = None

        # Make boolean
        boolean_predictions = (y_predicted >= .5).astype(int)

        if print_classification_report:
            print(classification_report(y_true, boolean_predictions))

        # Brier score
        brier = brier_score_loss(y_true, y_predicted)
        print('Brier score', brier)
        if score_to_return == 'brier':
            score = brier

        # AUC
        roc_auc = roc_auc_score(y_true, y_predicted)
        roc_auc = round(roc_auc, 4 - int(np.floor(np.log10(abs(roc_auc)))) - 1)
        if score_to_return == 'roc_auc':
            score = roc_auc
        if plot_roc:
            f_roc, ax_roc = plt.subplots()
            fpr, tpr, thresholds = roc_curve(y_true, y_predicted)
            ax_roc.plot(fpr, tpr)
            # plot no skill line
            ax_roc.plot([0, 1], [0, 1], linestyle='--')

            ax_roc.set_title(f"ROC AUC {roc_auc}")
            ax_roc.set_xlabel("1-Specificity (False Pos Rate)")
            ax_roc.set_ylabel("Sensitivity (True Pos Rate)")

        # Plot confusion matrix
        if plot_confusion_matrix:
            conf_mat = confusion_matrix(y_true, boolean_predictions, normalize='true')
            f, ax = plt.subplots(figsize=(9, 7))
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)  # , display_labels=my_classifier.classes_)
            disp.plot(ax=ax)

        if show_now:
            plt.show()

        return score

    def evaluate_regressor_on_test_data(self):
        """

        :return:
        """
        raise NotImplementedError
