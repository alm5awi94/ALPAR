"""Code to work with a custom json config-format (reading, validation, ...)."""

import json
import logging
import os
from typing import Dict, List, Tuple, Union

from data.data import Data

logger = logging.getLogger(__name__)


# Classes for easy access to and renaming of dictionary keys.
# Useful with autocompletion and also for preventing KeyErrors.


class ConfKeys:
    """Top-level keys used in the configuration file."""

    GLOBAL = "global"
    DATA = "data"
    MODEL = "model"
    EXTRACTION = "extraction"
    EVALUATION = "evaluation"


class GlobalKeys:
    """Keys used in the global configuration."""

    LOGGING = "logging"
    SEED = "seed"
    METRICS_FILENAME = "metrics_filename"
    RULES_FILENAME = "rules_filename"
    PREDICT_INSTANCE_FILENAME = "predict_instance_filename"


class DataKeys:
    """Keys used in the data module configuration."""

    INPUT_PATH = "input_path"
    OUTPUT_PATH = "output_path"
    DATASET_NAME = "dataset_name"
    ORIG_SHAPE = "original_shape"
    TEST_SIZE = "test_size"
    LABEL_COL = "label_col"
    CAT_CONV_METHOD = "cat_conv_method"
    CATEGORICAL_COLUMNS = "categorical_columns"
    SCALE_DATA = "scale_data"


class ModelKeys:
    """Keys used in the model module configuration."""

    OUTPUT_PATH = "output_path"
    DATA_PATH = "data_path"
    TYPE = "nwtype"
    HIDDEN_LAYERS = "hidden_layer_units"
    HIDDEN_LAYER_ACTIVATIONS = "hidden_layer_activations"
    CONV_LAYER_KERNELS = "conv_layer_kernels"
    USE_CLASS_WEIGHTS = "use_class_weights"
    BATCH_SIZE = "batch_size"
    EPOCHS = "epochs"
    VAL_SPLIT = "val_split"
    DROPOUT = "dropout"
    LEARNING_RATE = "learning_rate"
    USE_DECAY = "use_decay"


class ExtractionKeys:
    """Keys used in the extraction module configuration."""
    MODEL_PATH = "trained_model_path"
    DATA_PATH = "data_path"
    ALGORITHM = "algorithm"
    OUTPUT_PATH = "rules_dir"


class EvaluationKeys:
    """Keys used in the evaluation module configuration."""
    MODEL_PATH = "trained_model_path"
    RULES_DIR = "rules_dir"
    DATA_PATH = "data_path"
    OUTPUT_PATH = "evaluation_dir"


class FileNames:
    """Common file names used in the project."""
    RULES = "rules.bin"
    METRICS = "eval_metrics.json"
    PREDICT_INSTANCE = "rule_classifier.pickle"


# contains all the possible keys with indication of their data type
# actually not used at the moment, so only here for documentation purposes
_template: dict = {
    "global": {
        GlobalKeys.LOGGING: "WARNING",
        GlobalKeys.SEED: 42,
        GlobalKeys.METRICS_FILENAME: FileNames.METRICS,
        GlobalKeys.RULES_FILENAME: FileNames.RULES
    },
    ConfKeys.DATA: {
        DataKeys.INPUT_PATH: "",
        DataKeys.OUTPUT_PATH: "",
        DataKeys.LABEL_COL: 0,
        DataKeys.ORIG_SHAPE: [],
        DataKeys.TEST_SIZE: 0.3,
        DataKeys.DATASET_NAME: "",
        DataKeys.CAT_CONV_METHOD: "woe",
        DataKeys.CATEGORICAL_COLUMNS: [],
        DataKeys.SCALE_DATA: False
    },
    ConfKeys.MODEL: {
        ModelKeys.DATA_PATH: "",
        ModelKeys.TYPE: "ff",
        ModelKeys.HIDDEN_LAYERS: [],
        ModelKeys.HIDDEN_LAYER_ACTIVATIONS: [],
        ModelKeys.CONV_LAYER_KERNELS: [],
        ModelKeys.USE_CLASS_WEIGHTS: True,
        ModelKeys.BATCH_SIZE: 1,
        ModelKeys.EPOCHS: "",
        ModelKeys.OUTPUT_PATH: "",
        ModelKeys.LEARNING_RATE: 0.001,
        ModelKeys.USE_DECAY: False,
        ModelKeys.DROPOUT: 0.5
    },
    ConfKeys.EXTRACTION: {
        ExtractionKeys.MODEL_PATH: "",
        ExtractionKeys.DATA_PATH: "",
        ExtractionKeys.ALGORITHM: "dnnre",
        ExtractionKeys.OUTPUT_PATH: ""
    },
    ConfKeys.EVALUATION: {
        EvaluationKeys.MODEL_PATH: "",
        EvaluationKeys.RULES_DIR: "",
        EvaluationKeys.DATA_PATH: "",
        EvaluationKeys.OUTPUT_PATH: ""
    },
}


class Configuration:
    def __init__(self, paths: Union[str, List[str]]):
        self._json_data: Dict = {}
        self.paths = paths
        if type(self.paths) is str:
            self.paths = [self.paths]
        self.read()

    def get_all(self) -> Dict:
        """:return: The raw data which was read."""
        return self._json_data

    def get_global_params(self) -> Dict:
        return self._json_data[ConfKeys.GLOBAL]

    def has_module(self, conf_key: str) -> bool:
        return conf_key in self._json_data

    def _get_module_params(self, conf_key: str) -> Dict:
        """
        Returns the second-level dict with module-specific keys.

        :param conf_key: The json-key of the module.

        :return: A dictionary containing the module specific values.

        """
        if conf_key in self._json_data:
            return self._json_data[conf_key]
        else:
            raise ValueError(
                f'The configuration is missing the required "{conf_key}" key.')

    def get_data_params(self) -> Dict:
        return self._get_module_params(ConfKeys.DATA)

    def get_model_params(self) -> Dict:
        return self._get_module_params(ConfKeys.MODEL)

    def get_extraction_params(self) -> Dict:
        return self._get_module_params(ConfKeys.EXTRACTION)

    def get_evaluation_params(self) -> Dict:
        return self._get_module_params(ConfKeys.EVALUATION)

    def validate_all(self) -> None:
        """Validates whole configuration."""
        self.validate_global()
        self.validate_data()
        self.validate_model()
        self.validate_evaluation()
        self.validate_rule_ex()

    def validate_model(self) -> None:
        """Validates the tensorflow model model configuration."""
        logger.debug("Validating configuration for model.")
        model_params = self.get_model_params()
        if EvaluationKeys.DATA_PATH not in model_params:
            raise ValueError(
                f"The required key {ModelKeys.DATA_PATH} is not in the "
                f"{ConfKeys.MODEL} section.")
        if ModelKeys.TYPE not in model_params:
            raise ValueError(
                f"The {ConfKeys.MODEL} configuration is missing the required "
                f"key {ModelKeys.TYPE}.")
        if not model_params[ModelKeys.TYPE] in ["ff", "conv"]:
            raise ValueError(
                f"The parameter {ModelKeys.TYPE} in {ConfKeys.MODEL} "
                f"has to be either 'ff' or 'conv'.")
        if not model_params[ModelKeys.BATCH_SIZE] > 0:
            raise ValueError(
                f"The parameter {ModelKeys.BATCH_SIZE} of {ConfKeys.MODEL} has"
                f" to be greater than 0.")
        if (
                not min(model_params[ModelKeys.HIDDEN_LAYERS]) > 0
                or type(model_params[ModelKeys.HIDDEN_LAYERS]) != list
                or (len(model_params[ModelKeys.HIDDEN_LAYERS]) > 0
                    and type(model_params[ModelKeys.HIDDEN_LAYERS][0]) != int)
        ):
            raise ValueError(
                f"The parameter {ModelKeys.HIDDEN_LAYERS} must be a (empty) "
                f"list of non-zero integers.")
        if ModelKeys.HIDDEN_LAYER_ACTIVATIONS not in model_params:
            self._json_data[ConfKeys.MODEL][
                ModelKeys.HIDDEN_LAYER_ACTIVATIONS] \
                = ["relu" for _ in model_params[ModelKeys.HIDDEN_LAYERS]]
        if (
                len(model_params[ModelKeys.HIDDEN_LAYERS]) !=
                len(model_params[ModelKeys.HIDDEN_LAYER_ACTIVATIONS])
        ):
            raise ValueError(
                f"The parameter {ModelKeys.HIDDEN_LAYER_ACTIVATIONS} must "
                f"specify a list of activations with one activation for each "
                f"hidden layer specified by {ModelKeys.HIDDEN_LAYERS}.")
        valid_activations = ["linear", "relu", "exponential", "sigmoid",
                             "softmax", "tanh"]
        for act in model_params[ModelKeys.HIDDEN_LAYER_ACTIVATIONS]:
            if act not in valid_activations:
                raise ValueError(
                    f"The activation functions specified in "
                    f"{ModelKeys.HIDDEN_LAYER_ACTIVATIONS} must be one of "
                    f"{valid_activations}.")
        if (
                model_params[ModelKeys.TYPE] == "conv" and
                ModelKeys.CONV_LAYER_KERNELS not in model_params
        ):
            raise ValueError(
                f"A model of type conv requires the key"
                f" {ModelKeys.CONV_LAYER_KERNELS}.")
        if model_params[ModelKeys.TYPE] == "conv" and (
                len(model_params[ModelKeys.CONV_LAYER_KERNELS]) !=
                len(model_params[ModelKeys.HIDDEN_LAYERS]) or
                model_params[ModelKeys.CONV_LAYER_KERNELS][0] == [] or
                len(model_params[ModelKeys.CONV_LAYER_KERNELS][0]) != 2):
            raise ValueError(
                f"The parameter {ModelKeys.CONV_LAYER_KERNELS} must be a "
                f"list of list of dimension 2 and have the same length as "
                f"{ModelKeys.HIDDEN_LAYERS}.")
        if ModelKeys.USE_CLASS_WEIGHTS not in model_params:
            self._json_data[ConfKeys.MODEL][ModelKeys.USE_CLASS_WEIGHTS] = True
        if (
                ModelKeys.LEARNING_RATE in model_params and
                not 0.0 <= model_params[ModelKeys.LEARNING_RATE] < 1.0
        ):
            raise ValueError(
                f"The parameter {ModelKeys.LEARNING_RATE} of {ConfKeys.MODEL}"
                f"has to be in [0.0, 1.0].")
        if (
                ModelKeys.LEARNING_RATE in model_params and
                not 0.0 <= model_params[ModelKeys.VAL_SPLIT] < 1.0
        ):
            raise ValueError(
                f"The parameter {ModelKeys.VAL_SPLIT} of {ConfKeys.MODEL}"
                f"has to be in [0.0, 1.0].")
        if (
                ModelKeys.DROPOUT in model_params and
                not 0.0 <= model_params[ModelKeys.DROPOUT] < 1.0
        ):
            raise ValueError(
                f"The parameter {ModelKeys.DROPOUT} of {ConfKeys.MODEL}"
                f"has to be in [0.0, 1.0].")

    def validate_rule_ex(self) -> None:
        """Validates the rule extraction configuration."""
        logger.debug("Validating configuration for rule extraction.")
        extrac_params = self.get_extraction_params()
        if EvaluationKeys.DATA_PATH not in extrac_params:
            raise ValueError(
                f"The required key {ExtractionKeys.DATA_PATH} is not in the "
                f"{ConfKeys.EXTRACTION} section.")
        valid_algorithms = ["dnnre", "alpa"]
        if extrac_params[ExtractionKeys.ALGORITHM] not in valid_algorithms:
            raise ValueError(
                f"The parameters {ExtractionKeys.ALGORITHM} of"
                f"{ConfKeys.EXTRACTION} must be "
                f"one of {valid_algorithms}")

    def validate_evaluation(self) -> None:
        """Validates the evaluation configuration."""
        logger.debug("Validating configuration for evaluation.")
        eval_params = self.get_evaluation_params()
        if EvaluationKeys.DATA_PATH not in eval_params:
            raise ValueError(
                f"The required key {EvaluationKeys.DATA_PATH} is not in the "
                f"{ConfKeys.EVALUATION} section.")

    def validate_global(self) -> None:
        """Validate the global configuration."""

        # global params validation
        if ConfKeys.GLOBAL not in self._json_data:
            raise ValueError(
                'The configuration is missing the '
                'required global configuration.')

        global_params = self.get_global_params()
        valid_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
        if global_params[GlobalKeys.LOGGING] not in valid_levels:
            raise ValueError(
                f"The parameter {GlobalKeys.LOGGING} of {ConfKeys.GLOBAL}"
                f"must be one of {valid_levels}.")
        logging.basicConfig(level=logging.getLevelName(
            global_params[GlobalKeys.LOGGING]))
        logger.debug("Validated configuration for global.\n"
                     "\t Validating configuration for data.")

    def validate_data(self) -> None:
        """Validate the data configuration."""
        data_params = self.get_data_params()
        if DataKeys.OUTPUT_PATH not in data_params:
            raise ValueError(f"The required key {DataKeys.OUTPUT_PATH} is "
                             f"not provided in the {ConfKeys.DATA} section.")
        if not 0.0 <= data_params[DataKeys.TEST_SIZE] < 1.0:
            raise ValueError(
                f"The parameter {DataKeys.TEST_SIZE} of {ConfKeys.DATA}"
                f"has to be in [0.0, 1.0].")
        if not os.path.isfile(data_params[DataKeys.INPUT_PATH]):
            raise ValueError(
                f"The data input path {data_params[DataKeys.INPUT_PATH]} "
                f"is not a valid file.")
        if DataKeys.CATEGORICAL_COLUMNS in data_params:
            only_integers = all([type(x) == int and x >= 0 for x in
                                 data_params[DataKeys.CATEGORICAL_COLUMNS]])
            only_strings = all([type(x) == str for x in
                                data_params[DataKeys.CATEGORICAL_COLUMNS]])
            if not (only_integers or only_strings):
                raise ValueError(
                    f"The parameter {DataKeys.CATEGORICAL_COLUMNS} "
                    f"of {ConfKeys.DATA} has to be a list of strings "
                    f"or non negative integers")
        if DataKeys.CAT_CONV_METHOD in data_params:
            methods = Data.cat_encoder_methods.keys()
            if data_params[DataKeys.CAT_CONV_METHOD] not in methods:
                raise ValueError(
                    f"The parameter {DataKeys.CAT_CONV_METHOD} "
                    f"of {ConfKeys.DATA} has to be one of "
                    f"{methods}")
        if DataKeys.SCALE_DATA in data_params:
            if type(data_params[DataKeys.SCALE_DATA]) != bool:
                raise ValueError(
                    f"The parameter {DataKeys.SCALE_DATA} "
                    f"of {ConfKeys.DATA} has to be a bool)")

    def read(self):
        """Tries to read the configuration from the path provided at
        initialisation into the internal ``_json_data`` attribute, which can be
        accessed using the other methods.
        """

        # used in json.load to raise an exception when there is a duplicate key
        def check_duplicate_key(pairs: List[Tuple]):
            """
            Checks whether there is a duplicate key in pairs.
            :param pairs: A list of key-values pairs
            :return: The pairs as dict.
            """
            keys = {}
            for k, v in pairs:
                if k in keys:
                    raise ValueError(
                        f"Configuration disallows duplicate keys ({k}). Use "
                        f"multiple configurations/CLI calls instead.")
                else:
                    keys[k] = v
            return keys

        # merge all the configs provided into one
        # when two configs contain the same key, the later config takes
        # precedence
        for path in self.paths:
            with open(path, "r") as file:
                self._json_data.update(
                    json.load(file, object_pairs_hook=check_duplicate_key))
        # Override defaults for the global config
        global_params = self.get_global_params()
        if GlobalKeys.LOGGING not in global_params:
            self._json_data[ConfKeys.GLOBAL][
                GlobalKeys.LOGGING] = "WARNING"
        if GlobalKeys.METRICS_FILENAME in global_params:
            FileNames.METRICS = global_params[GlobalKeys.METRICS_FILENAME]
        if GlobalKeys.RULES_FILENAME in global_params:
            FileNames.RULES = global_params[GlobalKeys.RULES_FILENAME]
        if GlobalKeys.PREDICT_INSTANCE_FILENAME in global_params:
            FileNames.PREDICT_INSTANCE = global_params[
                GlobalKeys.PREDICT_INSTANCE_FILENAME]

    def write(self, output_path: str):
        """Writes the current configuration to `output_path`."""
        with open(output_path, "w+") as file:
            json.dump(self._json_data, file)
