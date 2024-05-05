from typing import Any, Tuple

import tensorflow as tf
from data_manager import DataManager
from flops_utils.ml_repo_templates import ModelManagerTemplate


class ModelManager(ModelManagerTemplate):
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2(
            (32, 32, 3), classes=10, weights=None
        )
        self.loss_function = "sparse_categorical_crossentropy"
        self.optimizer = "adam"
        self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

        # self.x_train = None
        # self.y_train = None
        # self.x_test = None
        # self.y_test = None

    def prepare_data(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (
            DataManager().get_data()
        )

    def get_model(self) -> Any:
        return self.model

    def get_model_parameters(self) -> Any:
        return self.model.get_weights()

    def set_model_parameters(self, parameters) -> None:
        self.model.set_weights(parameters)

    def fit_model(self) -> int:
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return len(self.x_train)

    def evaluate_model(self) -> Tuple[Any, Any, int]:
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, accuracy, len(self.x_test)
