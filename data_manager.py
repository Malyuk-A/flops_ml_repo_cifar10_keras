from typing import Any, Tuple

import tensorflow as tf


class DataManager:
    def __init__(self):
        self.training_data, self.testing_data = tf.keras.datasets.cifar10.load_data()

    def get_data(self) -> Tuple[Any, Any]:
        return self.training_data, self.testing_data
