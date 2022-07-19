import logging
import unittest

from rea.data.data import Data
from rea.model.model import Model

logging.basicConfig(level=logging.DEBUG)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path_iris = "temp/model_unit/datairis"
        self.data_ff = Data("resources/iris.csv", self.data_path_iris,
                            dataset_name="Iris", label_col="Species")
        self.data_path_mnist = "temp/model_unit/datamnist"
        self.data_conv = Data("resources/mnist_30.csv", self.data_path_mnist,
                              dataset_name="MNIST",
                              label_col=0, original_shape=[28, 28, 1])

    def test_run_default_ff(self):
        model = Model(nwtype="ff", output_path="temp/model_unit",
                      data_path=self.data_path_iris,
                      hidden_layer_units=[10, 50],
                      hidden_layer_activations=["linear", "relu"], epochs=1)
        model.run(self.data_ff.run())
        model.run()

    def test_run_default_conv(self):
        model = Model(nwtype="conv", output_path="temp/model_unit",
                      data_path=self.data_path_mnist,
                      hidden_layer_units=[10, 50],
                      hidden_layer_activations=["relu", "relu"],
                      conv_layer_kernels=[[3, 3], [3, 3]], epochs=1)
        model.run(self.data_conv.run())
        model.run()

    def test_create_ff_model(self):
        # TODO: implement
        pass


if __name__ == "__main__":
    unittest.main()
