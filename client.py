from model_manager import ModelManager
from data_manager import DataManager

class Client:
    def __init__(self):
        self.model = ModelManager().get_model()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = DataManager().get_data()

    def get_model(self):
        return self.model

    def get_model_parameters(self):
        return self.model.get_weights()

    def set_model_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit_model(self):
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return len(self.x_train)

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, accuracy, len(self.x_test)



