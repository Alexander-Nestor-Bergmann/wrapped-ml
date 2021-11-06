# Welcome to wrapped-ml

wrapped-ml is a machine learning library that wraps Keras, 
Scikit-learn and XGBoost into a common framework.

N.B. this package is in development elsewhere and updates 
will be posted to this repository sparingly, as they finish.
In the meantime, feel free to use what's here!
<hr/>

## Usage

Some brief example usage for a classifier:

```python
# Import type of model you want to use e.g. ClassifierNNClass, ClassifierSklearnClass or custom (as here).
from ExampleNNClassifierClass import ExampleNNClassifierClass

# We will build NN Classifier, let's specify the architecture as a list of layer names and their params
network_architecture: list = [('Dense', {'units': 8, 'activation': 'relu'}),
                              ('Dropout', {'rate': 0.1}),
                              ('Dense', {'units': 1, 'activation': 'sigmoid'})]
# Build params that are passed to the base model (Note, you'll also have to specify input shape)
network_input_data: dict = {'network_architecture': network_architecture}
# This example class takes a string as a dummy example custom input
dummy_input: str = "test"

# Build the classifier
my_classifier = ExampleNNClassifierClass(dummy_input, network_input_data)

# Build and train the model using a test and train Dataframe
my_classifier.train_model(train_df=some_train_df, test_df=some_test_df)

# Optionally calibrate probabilties
my_classifier.calibrate_probabilities()

# Evaluate on some validation data, with orptional parameters in Dict: testing_func_args
my_classifier.evaluate_on_test_data(validation_input, validation_output, testing_args=testing_func_args)

```

## Dependencies

- Python 3.x
- dill~=0.3.3
- joblib~=1.0.1
- matplotlib~=3.4.2
- numpy~=1.19.5
- scikit-learn~=0.24.2
- tensorflow~=2.5.0
- xgboost~=1.4.2