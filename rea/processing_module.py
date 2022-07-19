import os
from abc import ABC
from typing import Optional

from data.data import Data


class ProcessingModule(ABC):
    """Base class for modules of the pipeline that use `Data`."""

    def __init__(
            self,
            output_dir: str,
            data_dir: str,
            seed: int,
    ):
        """
        Common attributes among Pipeline modules.

        :param output_dir: path where to save the outputs
        :param data_dir: path to the `Data` output folder
        :param seed: random seed to use

        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_dir = data_dir
        self.data: Optional[Data] = None
        self.seed = seed

    def setup_data(self, data: Optional[Data] = None) -> None:
        """
        Sets the data attribute to data. If None is provided
        a `Data` instance is created with `Data.read`.

        :param data: Instance to use or None.

        """
        if data is None:
            self.data = Data.read(self.data_dir)
        else:
            self.data = data

    def run(self, data: Data = None) -> None:
        """
        Execute the primary module function.

        :param data: (Optional) Can be provided when using the API mode.

        """
        raise NotImplementedError
