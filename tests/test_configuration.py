import logging
import unittest

from rea.configuration import Configuration

logging.basicConfig(level=logging.DEBUG)


class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.test_file = "resources/test_config_dnnre.json"
        self.test_file_conv = "resources/test_config_conv.json"
        self.test_file_minimal = "resources/test_config_minimal.json"
        self.test_file_wrong = "resources/test_config_wrong.json"

    def test_init(self):
        Configuration(self.test_file).validate_all()
        Configuration(self.test_file_minimal)
        Configuration(self.test_file_conv).validate_all()
        with self.assertRaises(ValueError):
            Configuration(self.test_file_wrong).validate_all()

    # TODO: add tests for validate
