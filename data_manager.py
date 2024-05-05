from typing import Any, Tuple

import tensorflow as tf
from flops_utils.ml_repo_templates import DataManagerTemplate


class DataManager(DataManagerTemplate):
    def __init__(self):
        self.training_data, self.testing_data = tf.keras.datasets.cifar10.load_data()

    def get_data(self) -> Tuple[Any, Any]:
        return self.training_data, self.testing_data
