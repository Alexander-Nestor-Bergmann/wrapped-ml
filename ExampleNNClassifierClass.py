#!/usr/bin/env python
# encoding: utf-8

from abc import ABC
from typing import Dict, List, Tuple

from src.ClassifierNNClass import ClassifierNNClass


class ExampleNNClassifierClass(ClassifierNNClass, ABC):
    """

    """

    def __init__(self, an_example_param: str, **base_kwargs):
        """

        To build a custom network architecture, pass default_network_architecture a list of layers. Each item in
        list is a Tuple: ('name_of_layer', layer_param_dict) where layer_param_dict is a Dict of compatible input
        params for the layer in Keras e.g. see below.

        :param base_kwargs:
        """
        super().__init__(**base_kwargs)

        self.default_network_architecture: List[Tuple[str, Dict]] = [('Dense', {'units': 8, 'activation': 'relu'}),
                                                                     ('Dropout', {'rate': 0.1}),
                                                                     ('Dense', {'units': 1, 'activation': 'sigmoid'})] \
            if base_kwargs.get('network_architecture', None) is None else base_kwargs.get('network_architecture')

        # Store custom inputs
        self.my_example_param = an_example_param

    #####

    def my_special_function(self, example_input: str = 'this is a test function') -> str:
        """
        Can add custom functions that are unique, or overload those of ClassifierNNClass
        """
        return f"{example_input} with example input param {self.my_example_param}"
