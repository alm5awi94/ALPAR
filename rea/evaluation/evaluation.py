import json
import logging
import os
import pickle
import random
from typing import Optional, Set

import numpy as np
import seaborn as sn
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from pandas import DataFrame
from rpy2.robjects.vectors import ListVector as R_c5
from tensorflow.keras.models import load_model

from data.data import Data
from evaluation.evaluate_rules.accuracy import accuracy
from evaluation.evaluate_rules.comprehensibility import comprehensibility
from evaluation.evaluate_rules.confusion_matrix import confusion_matrix
from evaluation.evaluate_rules.fidelity import fidelity
from evaluation.evaluate_rules.overlapping_features import overlapping_features
from evaluation.evaluate_rules.predict import predict as dnnre_predict
from configuration import FileNames
from extraction.alpa.alpa_c5 import c5_r_predict
from processing_module import ProcessingModule
from rules.helpers import neuron_to_str, pretty_string_repr
from rules.term import Neuron

logger = logging.getLogger(__name__)


class Evaluation(ProcessingModule):
    """Module for evaluating extracted rules."""

    def __init__(self, trained_model_path: str, data_path: str, rules_dir: str,
                 evaluation_dir: str, seed: int = 42):
        """
        Evaluate the extracted rules.

        :param trained_model_path: path to the trained tensorflow model
        :param data_path: path to the `Data` output folder
        :param rules_dir: path to the rules folder
        :param evaluation_dir: path of folder to save results
        :param seed: random seed for evaluation

        """
        super().__init__(evaluation_dir, data_path, seed)
        random.seed(self.seed)  # for random.choice in predict.py
        logger.debug(
            f"Created new Evaluation instance. "
            f"Loading model from {trained_model_path}.")
        self.trained_model = load_model(trained_model_path)
        self.metrics = self.read_metrics(rules_dir)
        self.rules = self.read_rules(rules_dir)
        self._predict_instance = self.load_predict_instance(rules_dir)
        logger.debug(
            f"Loaded rules and metrics from {rules_dir}.")
        self.conf_matrices_dir = os.path.join(self.output_dir,
                                              "confusion_matrices")
        os.makedirs(self.conf_matrices_dir, exist_ok=True)

    def run(self, data: Data = None) -> None:
        """
        Execute the primary module function.

        :param data: (Optional) Can be provided when using the API mode.

        """
        self.setup_data(data)
        logger.debug("Starting default evaluation.")
        comprehensibility_results = comprehensibility(self.rules)
        common_features = overlapping_features(self.rules)

        # convert lists of neurons to string representations
        def cvt(x: Neuron) -> str:
            return neuron_to_str(x, self.data)

        common_features["feature_union_per_class"] = [
            ", ".join(map(cvt, x)) for x in
            common_features["feature_union_per_class"].values()
        ]
        common_features["feature_intersection_per_class"] = [
            ", ".join(map(cvt, x)) for x in
            common_features["feature_intersection_per_class"].values()
        ]
        common_features["feature_intersec"] = ", ".join(map(
            cvt,
            common_features["feature_intersec"])
        )
        for (x, y, prefix) in ((self.data.x_train, self.data.y_train, "train"),
                               (self.data.x_test, self.data.y_test, "test")):
            results = self._predict_and_evaluate(x, y)
            results.update(common_features)
            results.update(comprehensibility_results)
            self._output_evaluation_results(results, prefix)
            logger.debug(f"Evaluated {prefix}.")

    @property
    def use_dnnre_predict(self) -> bool:
        """Whether to use dnnre rule predict method or not."""
        return self._predict_instance is None

    def _predict_and_evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        results = dict()
        true_labels = self.data.inverse_transform_classes(np.argmax(y, axis=1))
        nn_labels = self.data.inverse_transform_classes(np.argmax(
            self.trained_model.predict(
                np.reshape(x, self.data.original_shape)),
            axis=1))

        if self.use_dnnre_predict:
            rules_labels = self.data.inverse_transform_classes(
                dnnre_predict(self.rules, x))
        else:
            rules_labels = self.data.inverse_transform_classes(
                c5_r_predict(self._predict_instance, x))

        results["rules_acc"] = accuracy(rules_labels, true_labels)
        results["rules_fid"] = fidelity(rules_labels, nn_labels)
        results["rules_conf_matrix"] = confusion_matrix(true_labels,
                                                        rules_labels)
        results["nn_conf_matrix"] = confusion_matrix(true_labels, nn_labels)
        results["nn_to_rules_conf_matrix"] = confusion_matrix(nn_labels,
                                                              rules_labels)
        _, nn_acc = self.trained_model.evaluate(
            np.reshape(x, self.data.original_shape),
            y)
        results["nn_acc"] = nn_acc
        return results

    def _output_evaluation_results(self, re_results: dict,
                                   dataset_kind: str) -> None:
        # define where directory of the template is, load environment with
        # that path and load eval_template.md as jinja-template
        templ_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'templates')

        env = Environment(loader=FileSystemLoader(templ_path))
        template = env.get_template('eval_templ.md')

        # dictionary 'res' for jinja.render() is re_results dictionary updated
        # by the rest of necessary values
        res = {}
        res.update(re_results)
        res.update(self.metrics)
        res["rules"] = self.rules
        res["human_rules"] = pretty_string_repr(self.rules, self.data)
        res["seed"] = self.seed
        res["dataset_kind"] = dataset_kind
        # write confusions matrices and save them as png to a given path,
        # if you fiddle with the path they are being saved to you also have to
        # change the path in eval_templ.md
        self.write_conf_matrix(
            re_results["rules_conf_matrix"],
            f"Confusion Matrix (Rules) on {dataset_kind} data",
            "Predicted by Rules", "Label in Dataset",
            self.conf_matrices_dir
        )
        self.write_conf_matrix(
            re_results["nn_conf_matrix"],
            f"Confusion Matrix (NN) on {dataset_kind} data",
            "Predicted by NN", "Label in Dataset",
            self.conf_matrices_dir
        )
        self.write_conf_matrix(
            re_results["nn_to_rules_conf_matrix"],
            f"Confusion Matrix (NN vs Rules) on {dataset_kind} data",
            "Predicted by Rules", "Predicted by NN",
            self.conf_matrices_dir
        )
        # Save the report as markdown file
        mdpath = os.path.join(self.output_dir,
                              f"{dataset_kind.lower()}_eval.md")
        with open(mdpath, "w+") as mdfile:
            mdfile.write(template.render(res))
        # Save the result dictionary as json for later algorithmic evaluation
        jsonpath = os.path.join(self.output_dir,
                                f"{dataset_kind.lower()}_eval.json")
        with open(jsonpath, "w+") as jsonfile:
            res["rules_conf_matrix"] = res[
                "rules_conf_matrix"].to_numpy().tolist()
            res["nn_conf_matrix"] = res["nn_conf_matrix"].to_numpy().tolist()
            res["nn_to_rules_conf_matrix"] = res[
                "nn_to_rules_conf_matrix"].to_numpy().tolist()
            del res["rules"]
            json.dump(res, jsonfile)

    @staticmethod
    def write_conf_matrix(conf_matrix: DataFrame, name: str, xlabel: str,
                          ylabel: str, conf_matrices_path: str):
        sn.heatmap(conf_matrix, cmap='inferno', annot=True, fmt='g')
        plt.title(name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cvt_name = name.replace("(", "").replace(")", "").replace(" ", "_")
        cvt_name = cvt_name.lower()
        output_path = os.path.join(conf_matrices_path, f'{cvt_name}.png')
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def read_metrics(rules_dir: str) -> dict:
        metrics_path = os.path.join(rules_dir, FileNames.METRICS)
        with open(metrics_path, mode='r') as file:
            json_dict: dict = json.load(file)
        return json_dict

    @staticmethod
    def read_rules(rules_dir: str) -> Set:
        rules_path = os.path.join(rules_dir, FileNames.RULES)
        with open(rules_path, 'rb') as rules_file:
            rules = pickle.load(rules_file)
        return rules

    @staticmethod
    def load_predict_instance(rules_dir: str) -> Optional[R_c5]:
        """
        Tries to load an instance of R `ListVector` for rule prediction.
        :param rules_dir: Directory to load from.
        :return: None if no instance found, otherwise rpy2 `ListVector`.
        """
        predict_instance_path = os.path.join(rules_dir,
                                             FileNames.PREDICT_INSTANCE)
        if os.path.isfile(predict_instance_path):
            logger.debug(
                f"Loading prediction instance from {predict_instance_path}.")
            with open(predict_instance_path, 'rb') as f:
                instance = pickle.load(f)
        else:
            instance = None
        return instance
