from typing import Any, Tuple

import tensorflow as tf

from data_manager import DataManager


class ModelManager:
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2(
            (32, 32, 3), classes=10, weights=None
        )
        self.loss_function = "sparse_categorical_crossentropy"
        self.optimizer = "adam"
        self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

        (self.x_train, self.y_train), (self.x_test, self.y_test) = (
            DataManager().get_data()
        )

    def get_model_parameters(self) -> Any:
        return self.model.get_weights()

    def set_model_parameters(self, parameters) -> None:
        self.model.set_weights(parameters)

    def fit_model(self) -> None:
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)

    def evaluate_model(self) -> Tuple[Any, Any]:
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, accuracy
