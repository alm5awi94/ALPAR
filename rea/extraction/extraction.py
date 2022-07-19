import json
import logging
import os
import pickle
import shutil
from time import process_time
from typing import Set, Tuple

import memory_profiler
from keras import models

from data.data import Data
from extraction.alpa.alpa import Alpa
from extraction.alpa.alpa_c5 import get_c5_rules

from extraction.dnnre.dnnre_misc import DataValues, get_output_classes
from extraction.dnnre.extract_rules.modified_deep_red_C5 import extract_rules as deep_red
from extraction.dnnre.model.trained_model import TrainedModel
from configuration import FileNames
from processing_module import ProcessingModule
from rules.rule import Rule

logger = logging.getLogger(__name__)


class Extraction(ProcessingModule):
    """
    Module for extracting rules from a trained feed forward neural network.
    """

    def __init__(
            self,
            trained_model_path: str,
            data_path: str,
            algorithm: str,
            rules_dir: str,
            seed: int = 42
    ):
        """
        Extract rules from a trained tensorflow model.

        :param trained_model_path: path where to load the model from
        :param data_path: path to the `Data` output folder
        :param algorithm: extraction algorithm to use
        :param rules_dir: path of folder to save rules and metrics
        :param seed: random seed for rule extraction algorithms

        """
        super().__init__(rules_dir, data_path, seed)
        self.trained_model_path = trained_model_path
        self.algorithm = algorithm
        self.metrics = {'time': 0.0, 'memory': 0.0}
        self.rules: Set = set()
        self.temp_path = os.path.join(self.output_dir, "temp")

    def run(self, data: Data = None) -> None:
        """
        Execute the primary module function.

        :param data: (Optional) Can be provided when using the API mode.

        """
        self.setup_data(data)
        os.makedirs(self.temp_path, exist_ok=True)
        start_time = process_time()
        start_memory = memory_profiler.memory_usage()[0]
        logger.debug(f"Starting rule extraction {self.algorithm}")
        if self.algorithm == "dnnre":
            self.rules = self._run_dnnre()
        elif self.algorithm == "alpa":
            self.rules, metrics = self._run_alpa()
            self.metrics.update(metrics)
        else:
            raise ValueError(
                f"Unknown rule extraction algorithm {self.algorithm}!")
        end_time = process_time()
        end_memory = memory_profiler.memory_usage()[0]
        self.metrics["algo"] = self.algorithm
        self.metrics['time'] = end_time - start_time
        self.metrics['memory'] = end_memory - start_memory
        logger.debug(f"Finished rule extraction {self.algorithm}\n"
                     f"Took {self.metrics['time']} sec "
                     f"and {self.metrics['memory']} MB of memory.")
        self._write_metrics()
        self._write_rules()
        self._clean_up()

    def _write_metrics(self):
        metrics_path = os.path.join(self.output_dir, FileNames.METRICS)
        with open(metrics_path, mode='w') as file:
            json.dump(self.metrics, file, indent=4)

    def _write_rules(self):
        rules_path = os.path.join(self.output_dir, FileNames.RULES)
        with open(rules_path, mode='wb') as file:
            pickle.dump(self.rules, file)
        logger.debug(f"Saved {len(self.rules)} rules to disc.")

    def _run_dnnre(
            self
    ) -> Set[Rule]:

        train_data = DataValues(X=self.data.x_train, y=self.data.y_train)
        test_data = DataValues(X=self.data.x_test, y=self.data.y_test)
        output_classes = get_output_classes(self.data.class_names)
        activations_path = os.path.join(self.temp_path, "activations")
        trained_model = TrainedModel(self.trained_model_path, output_classes,
                                     train_data, test_data, activations_path)
        return deep_red(trained_model, seed=self.seed)

    def _run_alpa(self) -> Tuple[Set[Rule], dict]:
        whitebox, metrics = Alpa(self.data).alpa(
            models.load_model(self.trained_model_path), self.seed)

        predict_instance_path = os.path.join(self.output_dir,
                                             FileNames.PREDICT_INSTANCE)
        with open(predict_instance_path, 'wb') as f:
            pickle.dump(whitebox, f)
        rules = get_c5_rules(whitebox)
        for rule in rules:
            logger.debug(rule)
        return rules, metrics

    def _clean_up(self) -> None:
        shutil.rmtree(self.temp_path)
