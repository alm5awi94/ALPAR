import logging
import unittest

from rpy2.robjects.vectors import ListVector as R_c5

from rea.data.data import Data
from rea.evaluation.evaluate_rules.predict import predict as dnnre_predict
from rea.evaluation.evaluation import Evaluation
from rea.extraction.alpa.alpa_c5 import c5_r_predict

logging.basicConfig(level=logging.DEBUG)


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path = "temp/eval_unit/data"
        self.data = Data("resources/iris.csv", self.data_path,
                         dataset_name="Iris",
                         label_col="Species")
        self.data.run()
        self.model_path = "resources/iris_model"
        self.rules_dir = "resources/iris_rules"
        self.rules = Evaluation.read_rules(self.rules_dir)
        self.evaluation_dir = "temp/eval_unit"

    def test_run_default(self):
        ev: Evaluation = Evaluation(self.model_path, self.data_path,
                                    self.rules_dir,
                                    self.evaluation_dir)
        ev.run(self.data)
        ev.run()

    def test_prediction(self):
        rule_classifier = Evaluation.load_predict_instance(self.rules_dir)
        self.assertTrue(type(rule_classifier) is R_c5)
        c5_predictions = c5_r_predict(rule_classifier, self.data.x_test)
        predictions = dnnre_predict(self.rules, self.data.x_test)
        self.assertEqual(len(c5_predictions), len(predictions))
        self.assertEqual(len(c5_predictions), len(self.data.y_test))


if __name__ == "__main__":
    unittest.main()
