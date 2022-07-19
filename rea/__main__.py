import argparse
from typing import Tuple

from rea import REA


def cli_parse_args() -> Tuple[str, Tuple[bool, bool, bool, bool]]:
    parser = argparse.ArgumentParser(
        prog="rea",
        description="Rule Extraction Assistant: Extracting rules from "
                    "artificial neural networks using different algorithms.",
        epilog="Per default, all modules specified in the provided config(s) "
               "are executed. You can limit the executed modules by using "
               "the optional arguments. If module arguments are passed there"
               "must be a corresponding configuration."
    )
    parser.add_argument("configuration", metavar="C", type=str, nargs='+',
                        help="Path to the configuration file. If multiple "
                             "paths are specified, the configurations are "
                             "merged into a single configuration.")
    parser.add_argument("-d", "--data", action="store_true",
                        help="Run the data module.")
    parser.add_argument("-t", "--train", action="store_true",
                        help="Run model module.")
    parser.add_argument("-r", "--ruleex", action="store_true",
                        help="Run rule extraction module.")
    parser.add_argument("-e", "--evaluate", action="store_true",
                        help="Run evaluation module.")
    args = parser.parse_args()
    return args.configuration, (
        args.data, args.train, args.ruleex, args.evaluate)


def main():
    conf_paths, flags = cli_parse_args()
    REA(conf_paths).run(*flags)


if __name__ == "__main__":
    main()
