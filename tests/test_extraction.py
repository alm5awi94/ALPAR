import logging
import unittest

import numpy as np
import tensorflow.keras.models as models

from rea.data.data import Data
from rea.evaluation.evaluate_rules.predict import predict
from rea.extraction.alpa.alpa import Alpa
from rea.extraction.alpa.alpa_c5 import c5_r_predict, get_c5_model, \
    get_c5_rules
from rea.extraction.extraction import Extraction

logging.basicConfig(level=logging.DEBUG)


class TestExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path = "temp/rule_unit/data"
        self.data = Data("resources/iris.csv", self.data_path,
                         dataset_name="Iris",
                         label_col="Species")
        self.data.run()
        self.alpa = Alpa(self.data)
        self.model_path = "resources/iris_model"
        self.model: models.Model = models.load_model(self.model_path)
        self.oracle_train = self.model.predict(self.data.x_train)
        self.rules_dir_dnnre = "temp/rule_unit/dnnre"
        self.rules_dir_alpa = "temp/rule_unit/alpa"
        self.Nv = 26
        self.Nr = 5

    def test_run_dnnre(self):
        ex: Extraction = Extraction(self.model_path, self.data_path, "dnnre",
                                    self.rules_dir_dnnre)
        ex.run(self.data)
        ex.run()

    def test_run_alpa(self):
        ex: Extraction = Extraction(self.model_path, self.data_path, "alpa",
                                    self.rules_dir_alpa)
        ex.run(self.data)
        ex.run()

    def test_alpa(self):
        whitebox, metrics = self.alpa.alpa(self.model)
        rules = get_c5_rules(whitebox)
        self.assertEqual(type(rules), set)
        self.assertEqual(type(metrics), dict)
        self.assertIn("best_rho", metrics)
        self.assertIn("max_fid", metrics)
        # at least one rule for each class
        found_classes = []
        for r in rules:
            found_classes.append(r.get_conclusion().name)
        self.assertEqual(set(found_classes), set(range(self.data.num_classes)))

    def test_alpa_c5(self):
        rule_model = get_c5_model(self.data.x_test,
                                  np.argmax(self.data.y_test, 1),
                                  42)
        rules = get_c5_rules(rule_model)
        pred_vanilla = predict(rules, self.data.x_test)
        pred_c5 = c5_r_predict(rule_model, self.data.x_test)
        self.assertTrue(np.equal(pred_c5, pred_vanilla).all())
        # print(pretty_string_repr(rules, self.data, True))

    def test_class_encoding(self):
        # dataset = pd.read_csv("resources/iris.csv")
        # dataset['Class'] = dataset['Species'].apply(
        # lambda i: "C" + str(i))
        # dataset.drop('Species', axis=1, inplace=True)
        # dataset = dataset.sample(frac=1)
        # dataset.to_csv("resources/iris_class_encoding.csv", index=False)
        data = Data("resources/iris_class_encoding.csv",
                    "temp/rule_unit/data_encoding",
                    dataset_name="Iris",
                    label_col='Class')
        data.run()
        y_train_original = data.inverse_transform_classes(
            np.argmax(data.y_train, 1))
        rule_model = get_c5_model(data.x_train,
                                  y_train_original,
                                  42)
        get_c5_rules(rule_model)

    def test_valleypoints(self):
        valleypoints, valleypoint_classes = self.alpa.generate_valleypoints(
            self.data.x_train, self.oracle_train)
        indices = []
        for i, x in enumerate(self.data.x_train):
            index = []
            for vp in valleypoints:
                if np.array_equal(x, vp):
                    index.append(i)
            self.assertTrue(len(index) <= 1)
            if len(index) == 1:
                indices += index
        self.assertEqual(len(indices), self.Nv)
        valley_classes = np.argmax(self.data.y_train[indices], axis=1)
        self.assertTrue(
            2 <= len(np.unique(valley_classes)) <= self.data.num_classes)

    def test_generate_points(self):
        valleypoints, valleypoint_classes = self.alpa.generate_valleypoints(
            self.data.x_train, self.oracle_train)
        nearest = Alpa(self.data).get_nearest(valleypoint_classes,
                                              valleypoints)
        # test generate points
        artificial = Alpa(self.data).generate_points(valleypoints, self.Nr,
                                                     nearest)
        self.assertEqual(len(artificial), self.Nr)
        self.model.predict(artificial)


if __name__ == "__main__":
    unittest.main()
