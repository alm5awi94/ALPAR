from collections import namedtuple
from typing import List, Tuple

from rules.rule import OutputClass

DatasetMetaData = namedtuple(
    'DatasetMetaData', 'name target_col output_classes n_inputs n_outputs')
DataValues = namedtuple('DataValues', 'X y')


def get_output_classes(class_names: List[str]) -> Tuple[OutputClass]:
    """Create OutputCLass instance for each occurring class in label_data."""
    return tuple(
        (OutputClass(name=class_name, encoding=index) for index, class_name in
         enumerate(class_names)))
