"""
REA - Rule Extraction Assistant

A pipeline for evaluating rule extraction algorithms for (feed-forward) neural
networks.

"""

import logging
from typing import List, Optional, Union

from configuration import ConfKeys, Configuration, GlobalKeys
from data.data import Data
from evaluation.evaluation import Evaluation
from extraction.extraction import Extraction
from model.model import Model

logger = logging.getLogger(__name__)


class REA:
    """
    Main class of the rule extraction assistant to load the configuration and
    run the pipeline.
    """

    def __init__(self, conf_paths: Union[str, List[str]]):
        self.conf = Configuration(conf_paths)

    def run(self, do_data_flag: bool = False, do_model_flag: bool = False,
            do_extraction_flag: bool = False, do_eval_flag: bool = False):
        """
        Executes the specified pipeline modules.
        The specification of which modules to run can be done by using the cli
        flags or gets inferred from the given configurations.

        :param do_data_flag: CLI flag to execute `Data` module.
        :param do_model_flag: CLI flag to execute `Model` module.
        :param do_extraction_flag: CLI flag to execute `Extraction` module.
        :param do_eval_flag: CLI flag to execute `Evaluation` module.

        """
        globl = self.conf.get_global_params()
        logging.basicConfig(level=logging.getLevelName(
            globl[GlobalKeys.LOGGING]))

        if any((
                do_data_flag, do_model_flag, do_extraction_flag,
                do_eval_flag)):
            # only execute specified modules
            # throw an error if cli flag is passed but config is missing
            do_data = self._cli_module_check(do_data_flag, ConfKeys.DATA)
            do_model = self._cli_module_check(do_model_flag, ConfKeys.MODEL)
            do_extraction = self._cli_module_check(do_extraction_flag,
                                                   ConfKeys.EXTRACTION)
            do_eval = self._cli_module_check(do_eval_flag, ConfKeys.EVALUATION)
        else:
            # only try to run the modules
            # that are actually specified in the config
            do_data = self.conf.has_module(ConfKeys.DATA)
            do_model = self.conf.has_module(ConfKeys.MODEL)
            do_extraction = self.conf.has_module(
                ConfKeys.EXTRACTION)
            do_eval = self.conf.has_module(
                ConfKeys.EVALUATION)

        # validate necessary configuration
        for flag, validator in (
                (do_data, self.conf.validate_data),
                (do_model, self.conf.validate_model),
                (do_extraction, self.conf.validate_rule_ex),
                (do_eval, self.conf.validate_evaluation)
        ):
            if flag:
                validator()

        data: Optional[Data] = None  # modules create instance if necessary
        if do_data:
            # load common data instance for modules
            data = Data(**self.conf.get_data_params(),
                        seed=globl[GlobalKeys.SEED])
            data.run()

        # run the pipeline
        if do_model:
            model: Model = Model(**self.conf.get_model_params(),
                                 seed=globl[GlobalKeys.SEED])
            model.run(data)

        if do_extraction:
            extraction: Extraction = Extraction(
                **self.conf.get_extraction_params(),
                seed=globl[GlobalKeys.SEED])
            extraction.run(data)

        if do_eval:
            evaluation: Evaluation = Evaluation(
                **self.conf.get_evaluation_params(),
                seed=globl[GlobalKeys.SEED])
            evaluation.run(data)

    def _cli_module_check(self, do_module_cli_flag: bool,
                          conf_key: str) -> bool:
        if do_module_cli_flag:
            if not self.conf.has_module(conf_key):
                raise ValueError(
                    f'CLI flag set to use {conf_key} module'
                    f'but no {conf_key} configuration found.')
            else:
                return True
        else:
            return False
