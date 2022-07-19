import logging
import unittest

import numpy as np
import pandas as pd

from rea.data.data import Data

logging.basicConfig(level=logging.DEBUG)


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path = "resources/iris.csv"
        self.output_path = "temp/data_unit"
        self.label_col_str = "Species"
        self.label_col_int = 4
        self.dataset_name = "Iris"
        self.dataset = pd.read_csv(self.data_path)
        self.test_size = 0.3
        # get number of dataset entries and columns
        self.n_entries = self.dataset.count()[0]
        self.n_columns = len(self.dataset.columns)

    def test_load(self):
        Data(self.data_path, self.output_path, self.label_col_str,
             self.dataset_name)

    def test_split_labels(self):
        x, y, _, _ = Data._split_labels(self.dataset, self.label_col_str)
        x1, y1, _, _ = Data._split_labels(self.dataset, self.label_col_int)
        # using int/str for label columns makes no difference
        self.assertTrue(np.array_equal(x, x1))
        self.assertTrue(np.array_equal(y, y1))
        # number of entries matches
        self.assertGreaterEqual(len(x), self.n_entries)
        self.assertEqual(len(y), self.n_entries)
        # data has all columns
        self.assertEqual(x.shape[1], self.n_columns - 1)
        # only one label
        self.assertEqual(len(y.shape), 1)

    def test_split_train(self):
        x, y, _, _ = Data._split_labels(self.dataset, self.label_col_str)
        x_tr, x_te, y_tr, y_te = Data._split_test(x, y, 42, self.test_size)
        self.assertEqual(len(x_te), self.test_size * len(x))
        self.assertEqual(len(x_tr), (1 - self.test_size) * len(x))
        self.assertEqual(len(y_te), self.test_size * len(y))
        self.assertEqual(len(y_tr), (1 - self.test_size) * len(y))
        self.assertEqual(len(y_tr), len(x_tr))
        self.assertEqual(len(y_te), len(x_te))
        # number of classes are equal
        self.assertEqual(np.unique(y_te).size, np.unique(y_tr).size)

    def test_one_hot_encode(self):
        _, labels, _, _ = Data._split_labels(self.dataset, self.label_col_str)
        d: Data = Data(self.data_path, self.output_path, self.label_col_str,
                       self.dataset_name)
        encoded_labels, _ = d._one_hot_encode_classes(
            labels, np.unique(labels).shape[0])
        self.assertEqual(len(encoded_labels.shape), 2)
        self.assertEqual(encoded_labels.shape[0], self.n_entries)
        self.assertEqual(np.min(encoded_labels), 0.0)
        self.assertEqual(np.max(encoded_labels), 1.0)

    def test_get_class_weights(self):
        x, y, _, _ = Data._split_labels(self.dataset,
                                        self.label_col_str)
        # iris classes are balanced
        weight_dict = Data._get_class_weights(y)
        for key in weight_dict:
            self.assertAlmostEqual(weight_dict[key], 1.0)
        # testing unbalanced labels
        y_pruned = y[:(len(y) // 3) * 2 + 5]
        weight_dict = Data._get_class_weights(y_pruned)
        self.assertEqual(weight_dict, {0: 0.7, 1: 0.7, 2: 7.0})

    def test_run_default(self):
        d: Data = Data(self.data_path, self.output_path, self.label_col_str,
                       self.dataset_name)
        d.run()

    def test_categorical(self):
        cat_columns = ["SEX"]
        cat_indices = [1]
        label_col = "default.payment.next.month"
        # df = pd.read_csv("resources/data_10_features.csv")
        # df.drop(columns=["EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2"],
        # inplace=True)
        # df.to_csv("resources/uci_test.csv", index=False)
        for cats in (cat_columns, cat_indices):
            for method in Data.cat_encoder_methods.keys():
                d: Data = Data("resources/uci_test.csv",
                               "temp/data_categorical",
                               label_col, "UCI_subset", cat_conv_method=method,
                               categorical_columns=cats)
                d.run()


if __name__ == "__main__":
    unittest.main()
