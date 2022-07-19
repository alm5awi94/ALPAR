import logging
import subprocess
import unittest

from rea.rea import REA

logging.basicConfig(level=logging.DEBUG)


class TestREA(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path_dnnre = "resources/test_config_dnnre.json"
        self.config_path_alpa = "resources/test_config_alpa.json"
        self.config_path_alpa_conv = "resources/test_config_conv.json"
        self.config_path_mnist = "resources/test_config_conv.json"
        self.config_path_encoding = "resources/test_config_encoding.json"

    def test_run(self):
        r = REA(self.config_path_dnnre)
        r.run()
        r = REA(self.config_path_alpa)
        r.run()

    def test_encoding(self):
        r = REA(self.config_path_encoding)
        r.run()

    @unittest.skip("Takes approximately an hour to complete")
    def test_performance(self):
        r = REA(self.config_path_alpa_conv)
        r.run()

    @unittest.skip("Only needed for CLI tests")
    def test_cli(self):
        subprocess.run(["python", "-m", "rea",
                        "resources/test_config_dnnre.json"], check=True)

    # @unittest.skip(
    #    "Fails sometimes if GPU (CUDA) is used and many tests run.")
    def test_batch(self):
        subprocess.run(["python", "-m", "rea",
                        "resources/batch/global.json",
                        "resources/batch/data.json"], check=True)
        subprocess.run(["python", "-m", "rea",
                        "resources/batch/global.json",
                        "resources/batch/model.json"], check=True)
        subprocess.run(["python", "-m", "rea",
                        "resources/batch/global.json",
                        "resources/batch/extract_eval_alpa.json"], check=True)
        subprocess.run(["python", "-m", "rea",
                        "resources/batch/global.json",
                        "resources/batch/extract_eval_dnnre.json"], check=True)

    @unittest.skip("Only needed for CLI tests")
    def test_faulty_config(self):
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(["python", "-m", "rea",
                            "resources/test_config_wrong.json"],
                           check=True)
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(["python", "-m", "rea", "-t",
                            "resources/test_config_wrong.json"],
                           check=True)


if __name__ == "__main__":
    unittest.main()
