from typing import Any, Tuple

from flops_utils.ml_repo_building_blocks import load_dataset
from flops_utils.ml_repo_templates import DataManagerTemplate


class DataManager(DataManagerTemplate):
    def __init__(self):
        (self.x_train, self.x_test), (self.y_train, self.y_test) = self._prepare_data()

    def _prepare_data(self) -> Any:  # TODO adjust
        dataset = load_dataset()
        dataset.set_format("numpy")

        x, y = dataset["img"], dataset["label"]
        train_split = int(0.8 * len(x))
        x_train, x_test = x[:train_split], x[train_split:]
        eval_split = int(0.8 * len(y))
        y_train, y_test = y[:eval_split], y[eval_split:]
        # Attention: Unlike the sklearn example the params here are ordered differently.
        # The original keras dataset returned (x_train, y_train), (x_test, y_test)
        # instead of (x_train, x_test), (y_train, y_test)
        return (x_train, y_train), (x_test, y_test)

    def get_data(self) -> Tuple[Any, Any]:
        return (self.x_train, self.x_test), (self.y_train, self.y_test)
