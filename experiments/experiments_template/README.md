# Template for an Experiment

This folder proposes a structure for experiments conducted with REA.
Experiments adhering to this structure can be executed by the run.sh script
located at the experiment folder root.

Also refer to `test/resources/batch` and `tests/test_rea.py` for more inspiration.

Generally, you should create one experiment for each hypothesis.

|      name | purpose                                                                                                             |
|----------:|:--------------------------------------------------------------------------------------------------------------------|
|  `run.sh` | Used to make (multiple) calls to the REA CLI                                                                        |
|  `config` | Contains all configuration files                                                                                    |
| `eval.py` | Python script for plotting and further evaluation based on the json data output of the evaluation module and others |

## Generated Folders

These are suggestions for the output file structure

|                          name | purpose                         |
|------------------------------:|:--------------------------------|
|                     `log.txt` | Log output of rea               |
|              `data/[dataset]` | Output of the data module       |
|             `model/[dataset]` | Output of the model module      |
| `rules/[algorithm]/[dataset]` | Output of the rule extraction   |
|  `eval/[algorithm]/[dataset]` | Output of the evaluation module |
