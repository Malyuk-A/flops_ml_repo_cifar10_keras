import tensorflow as tf

class ModelManager:
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        self.loss_function = "sparse_categorical_crossentropy"
        self.optimizer = "adam"
        self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model
